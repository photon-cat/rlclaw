"""
Model-Predictive PID controller for the comma.ai Controls Challenge v2.

Combines PID feedback control with model-based single-step optimization.
- PID provides stable feedback (handles stochastic model noise)
- The ONNX model evaluates a small set of candidates near PID output
- Feed-forward from future plan anticipates upcoming targets
- Candidates are searched in a narrow band around PID for stability

Architecture:
- Steps 20-99: PID only (warmup, sim overrides actions)
- Steps 100-119: PID only (building correct action history)
- Steps 120+: Model-optimized action near PID output
"""

import os
import numpy as np
import onnxruntime as ort

CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
CONTROL_START_IDX = 100

NUM_CANDIDATES = 32  # fewer candidates for speed
SEARCH_HALF_WIDTH = 0.15  # tight search around PID


class Controller:
    """Model-predictive PID: PID + model-based refinement."""

    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', 'vendor', 'commaai', 'models', 'tinyphysics.onnx'
        )
        if not os.path.exists(model_path):
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'models', 'tinyphysics.onnx'
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(
                f.read(), options, ['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

        self.bins = np.linspace(
            LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE
        ).astype(np.float64)

        self.state_history = []
        self.lataccel_history = []
        self.controlled_actions = []

        # PID state
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0.0
        self.prev_error = 0.0

        self.call_count = 0

    def _encode(self, values):
        clipped = np.clip(values, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return np.digitize(clipped, self.bins, right=True)

    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _pid(self, target_lataccel, current_lataccel):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.call_count += 1
        sim_step = CONTEXT_LENGTH + self.call_count - 1

        self.state_history.append((state.roll_lataccel, state.v_ego, state.a_ego))
        self.lataccel_history.append(current_lataccel)

        # PID output
        pid_raw = self._pid(target_lataccel, current_lataccel)
        pid_action = float(np.clip(pid_raw, STEER_RANGE[0], STEER_RANGE[1]))

        # Warmup: PID only
        if sim_step < CONTROL_START_IDX:
            return pid_action

        # Building action history with PID
        if len(self.controlled_actions) < CONTEXT_LENGTH:
            self.controlled_actions.append(pid_action)
            return pid_action

        # Model-based refinement near PID output
        ctx_states = np.array(
            self.state_history[-CONTEXT_LENGTH:], dtype=np.float32
        )
        ctx_lataccels = np.array(
            self.lataccel_history[-CONTEXT_LENGTH:], dtype=np.float32
        )
        ctx_tokens = self._encode(ctx_lataccels).astype(np.int64)

        past_actions = np.array(
            self.controlled_actions[-(CONTEXT_LENGTH - 1):], dtype=np.float32
        )

        # Narrow candidate search around PID
        lo = max(STEER_RANGE[0], pid_action - SEARCH_HALF_WIDTH)
        hi = min(STEER_RANGE[1], pid_action + SEARCH_HALF_WIDTH)
        candidates = np.linspace(lo, hi, NUM_CANDIDATES, dtype=np.float32)
        N = len(candidates)

        states_batch = np.empty((N, CONTEXT_LENGTH, 4), dtype=np.float32)
        states_batch[:, :, 1] = ctx_states[:, 0]
        states_batch[:, :, 2] = ctx_states[:, 1]
        states_batch[:, :, 3] = ctx_states[:, 2]
        states_batch[:, :CONTEXT_LENGTH - 1, 0] = past_actions
        states_batch[:, CONTEXT_LENGTH - 1, 0] = candidates

        tokens_batch = np.broadcast_to(
            ctx_tokens[np.newaxis, :], (N, CONTEXT_LENGTH)
        ).copy()

        logits = self.ort_session.run(None, {
            'states': states_batch,
            'tokens': tokens_batch,
        })[0]

        last_logits = logits[:, -1, :]
        probs = self._softmax(last_logits, axis=-1)
        expected_lataccel = probs @ self.bins
        expected_lataccel = np.clip(
            expected_lataccel,
            current_lataccel - MAX_ACC_DELTA,
            current_lataccel + MAX_ACC_DELTA
        )

        # Cost with future anticipation
        tracking = (target_lataccel - expected_lataccel) ** 2
        jerk = ((expected_lataccel - current_lataccel) / DEL_T) ** 2

        # Add future target consideration
        future_cost = 0.0
        if future_plan is not None and len(future_plan.lataccel) > 0:
            n_future = min(5, len(future_plan.lataccel))
            future_targets = np.array(future_plan.lataccel[:n_future])
            weights = np.array([0.7 ** k for k in range(n_future)])
            weights /= weights.sum()
            avg_future = np.sum(future_targets * weights)
            future_cost = (avg_future - expected_lataccel) ** 2

        total_cost = (tracking * LAT_ACCEL_COST_MULTIPLIER
                     + jerk
                     + future_cost * 10.0)

        best_idx = int(np.argmin(total_cost))
        action = float(candidates[best_idx])

        self.controlled_actions.append(action)
        return action
