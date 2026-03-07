/**
 * Agent definitions — kept as a reference for multi-agent mode.
 * Currently unused; single-worker mode is defined inline in index.ts.
 * To switch to multi-agent, import these in index.ts and pass as `agents`.
 */

import type { AgentDefinition } from "@anthropic-ai/claude-agent-sdk";

const sharedTools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"] as const;

const CHALLENGE_CONTEXT = `
=== COMMA CONTROLS CHALLENGE ===
Goal: Design a controller for lateral car control that minimizes:
  total_cost = (lataccel_cost * 50) + jerk_cost

The simulator (tinyphysics.py) is an autoregressive ONNX model trained on real driving data.
Controller interface:
  class Controller(BaseController):
    def update(self, target_lataccel, current_lataccel, state, future_plan) -> float:
      # state: (roll_lataccel, v_ego, a_ego)
      # future_plan: (lataccel[50], roll_lataccel[50], v_ego[50], a_ego[50])
      # return: steer_action in [-2, 2]

Baseline PID scores ~85. The tfpgh solution scores 43.776.
GPU: Local RTX 5070 Ti (16GB VRAM). 15 min per experiment.
Reference code: vendor/commaai/ (original challenge), vendor/tfpgh/ (best solution)
Our controllers go in: src/controllers/
`;

export const agents: Record<string, AgentDefinition> = {
  "arch-search": {
    description:
      "Explores novel controller architectures: small MLPs, state-space models, attention-based controllers, hybrid PID+NN approaches.",
    prompt: `You are an ML architecture researcher specializing in compute-efficient models for control.
${CHALLENGE_CONTEXT}
Design small, fast controller architectures (<100K params). Write controllers to src/controllers/ and training code to src/algos/.`,
    tools: [...sharedTools],
  },

  "reward-optimizer": {
    description:
      "Optimizes loss functions, reward shaping, and training objectives for the controls challenge.",
    prompt: `You are an expert in loss function design and optimization for control problems.
${CHALLENGE_CONTEXT}
Design loss functions that directly minimize total_cost. Write to src/rewards/ and src/algos/configs/.`,
    tools: [...sharedTools],
  },

  "data-engineer": {
    description:
      "Manages training data: generates trajectories, builds datasets, handles the simulation pipeline.",
    prompt: `You are a data engineer for RL and behavioral cloning pipelines.
${CHALLENGE_CONTEXT}
Set up data generation, efficient loading, and explore cheap alternatives to PGTO teacher data.`,
    tools: [...sharedTools],
  },

  evaluator: {
    description:
      "Runs evaluations, compares controllers, generates reports, and tracks experiment results.",
    prompt: `You are the experiment evaluator and benchmarking lead.
${CHALLENGE_CONTEXT}
Run evals, track results in src/eval/results.json, generate comparison reports.`,
    tools: [...sharedTools],
  },

  "gpu-manager": {
    description:
      "Manages local GPU experiments: runs training scripts, monitors VRAM usage, collects results.",
    prompt: `You are the GPU infrastructure manager.
${CHALLENGE_CONTEXT}
Run training scripts on the local RTX 5070 Ti. Monitor GPU utilization and VRAM. Kill stuck processes.`,
    tools: [...sharedTools],
  },
};
