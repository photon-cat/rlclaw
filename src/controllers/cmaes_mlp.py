"""
CMA-ES optimized MLP controller for the comma.ai Controls Challenge.

Architecture: 14 inputs -> 20 hidden (tanh) -> 16 hidden (tanh) -> 1 output (tanh * 2.0)
Total parameters: 14*20+20 + 20*16+16 + 16*1+1 = 280+20+320+16+16+1 = 653
"""
import numpy as np
from vendor.commaai.controllers import BaseController


# Layer sizes
INPUT_DIM = 14
HIDDEN1_DIM = 20
HIDDEN2_DIM = 16
OUTPUT_DIM = 1
TOTAL_PARAMS = (INPUT_DIM * HIDDEN1_DIM + HIDDEN1_DIM +
                HIDDEN1_DIM * HIDDEN2_DIM + HIDDEN2_DIM +
                HIDDEN2_DIM * OUTPUT_DIM + OUTPUT_DIM)  # 653


def _mlp_forward(x, params):
    """Forward pass through the MLP using a flat parameter vector."""
    idx = 0

    # Layer 1: input -> hidden1
    w1 = params[idx:idx + INPUT_DIM * HIDDEN1_DIM].reshape(INPUT_DIM, HIDDEN1_DIM)
    idx += INPUT_DIM * HIDDEN1_DIM
    b1 = params[idx:idx + HIDDEN1_DIM]
    idx += HIDDEN1_DIM

    # Layer 2: hidden1 -> hidden2
    w2 = params[idx:idx + HIDDEN1_DIM * HIDDEN2_DIM].reshape(HIDDEN1_DIM, HIDDEN2_DIM)
    idx += HIDDEN1_DIM * HIDDEN2_DIM
    b2 = params[idx:idx + HIDDEN2_DIM]
    idx += HIDDEN2_DIM

    # Layer 3: hidden2 -> output
    w3 = params[idx:idx + HIDDEN2_DIM * OUTPUT_DIM].reshape(HIDDEN2_DIM, OUTPUT_DIM)
    idx += HIDDEN2_DIM * OUTPUT_DIM
    b3 = params[idx:idx + OUTPUT_DIM]

    h1 = np.tanh(x @ w1 + b1)
    h2 = np.tanh(h1 @ w2 + b2)
    out = np.tanh(h2 @ w3 + b3)
    return out[0] * 2.0  # scale to [-2, 2]


class Controller(BaseController):
    def __init__(self):
        super().__init__()
        self.params = np.zeros(TOTAL_PARAMS, dtype=np.float64)
        # State
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_actions = [0.0, 0.0]  # [t-1, t-2]

    @classmethod
    def from_params(cls, flat_array):
        """Create a controller from a flat numpy parameter vector."""
        ctrl = cls()
        ctrl.set_params(flat_array)
        return ctrl

    def set_params(self, flat_array):
        """Set MLP parameters from a flat numpy array."""
        assert len(flat_array) == TOTAL_PARAMS, f"Expected {TOTAL_PARAMS} params, got {len(flat_array)}"
        self.params = np.array(flat_array, dtype=np.float64)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Compute error terms
        error = target_lataccel - current_lataccel
        error_derivative = error - self.prev_error
        self.error_integral = np.clip(self.error_integral + error, -10.0, 10.0)

        # Future plan features (handle short/empty plans gracefully)
        fl = future_plan.lataccel if future_plan.lataccel else [0.0]
        fl_arr = np.array(fl, dtype=np.float64)
        n = len(fl_arr)

        future_near = np.mean(fl_arr[:min(5, n)]) if n > 0 else 0.0
        future_mid = np.mean(fl_arr[5:15]) if n > 5 else (np.mean(fl_arr) if n > 0 else 0.0)
        future_far = np.mean(fl_arr[15:40]) if n > 15 else (np.mean(fl_arr) if n > 0 else 0.0)
        sl = fl_arr[:min(40, n)]
        future_var = np.std(sl) if len(sl) > 0 else 0.0
        future_max_abs = np.max(np.abs(sl)) if len(sl) > 0 else 0.0

        # Build 14-dim input vector (manually normalized)
        x = np.array([
            error / 5.0,
            error_derivative / 2.0,
            self.error_integral / 5.0,
            target_lataccel / 5.0,
            state.v_ego / 30.0,
            state.a_ego / 4.0,
            state.roll_lataccel / 2.0,
            self.prev_actions[0] / 2.0,
            self.prev_actions[1] / 2.0,
            future_near / 5.0,
            future_mid / 5.0,
            future_far / 5.0,
            future_var / 5.0,
            future_max_abs / 5.0,
        ], dtype=np.float64)

        action = _mlp_forward(x, self.params)
        action = float(np.clip(action, -2.0, 2.0))

        # Update state
        self.prev_error = error
        self.prev_actions[1] = self.prev_actions[0]
        self.prev_actions[0] = action

        return action
