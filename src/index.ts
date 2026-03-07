import { query } from "@anthropic-ai/claude-agent-sdk";
import type { AgentDefinition } from "@anthropic-ai/claude-agent-sdk";

// Allow running from within another Claude Code session
delete process.env.CLAUDECODE;

const CHALLENGE_CONTEXT = `=== COMMA CONTROLS CHALLENGE ===
Goal: Minimize total_cost = (lataccel_cost * 50) + jerk_cost for lateral car control.
  - lataccel_cost: MSE between actual and target lateral acceleration
  - jerk_cost: smoothness penalty on lateral acceleration changes

Simulator: vendor/commaai/tinyphysics.py (autoregressive ONNX model, real driving data)
Controller interface:
  class Controller(BaseController):
    def update(self, target_lataccel, current_lataccel, state, future_plan) -> float:
      # state: (roll_lataccel, v_ego, a_ego)
      # future_plan: (lataccel[50], roll_lataccel[50], v_ego[50], a_ego[50])
      # return: steer_action in [-2, 2]

Scores: PID baseline ~85 (100 segs) | SOTA tfpgh 43.776
  tfpgh approach: CMA-ES MLP (~55) → GPU trajectory optimization (~43.2) → behavioral cloning student (43.776)

Reference code: vendor/commaai/ (challenge), vendor/tfpgh/ (best solution)
Our controllers: src/controllers/ | Training: src/algos/ | Results: src/eval/results.json

GPU: Local RTX 5070 Ti (16GB VRAM). Run experiments directly as python scripts.
Max 15 min per experiment. Controllers must run at 10Hz+, target <100K params.

Quick eval (~7s): cd vendor/commaai && python3 tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid`;

const worker: Record<string, AgentDefinition> = {
  worker: {
    description:
      "Executes research tasks: reads code, writes controllers, runs training, evaluates results. Use this for any task that requires tools.",
    prompt: `You are a research engineer working on the comma.ai Controls Challenge.
You receive specific tasks from the lead researcher and execute them thoroughly.

${CHALLENGE_CONTEXT}

You have access to all tools: read/write/edit files, run bash commands, search code.
Execute the task you're given completely, then report back with concrete results and metrics.
Be thorough but efficient. Always report numbers, not just "it worked".`,
    tools: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
  },
};

const ORCHESTRATOR_PROMPT = `You are the lead researcher for the comma.ai Controls Challenge.

${CHALLENGE_CONTEXT}

=== HOW YOU WORK ===
You have ONE worker agent ("worker") that executes tasks for you.
You PLAN what to do, then delegate ONE task at a time to the worker using the Agent tool.
When the worker returns, you REVIEW results, UPDATE your plan, and delegate the next task.

Each time you call the worker, give it a SPECIFIC, ACTIONABLE task. Not vague goals.
Good: "Read vendor/tfpgh/controllers/bc.py and vendor/tfpgh/offline/config.py, summarize the architecture, loss function, and training setup"
Bad: "Study the SOTA solution"

Good: "Write a CMA-ES training script to src/algos/cmaes_mlp.py that evolves a 2-hidden-layer MLP (32,16) to minimize total_cost on 100 segments. Run it for 5 minutes."
Bad: "Try to beat PID"

=== RESEARCH STRATEGY ===
Phase 1: Understand
  - Run PID baseline, study tfpgh solution architecture
  - Establish fast local eval pipeline
Phase 2: Quick wins
  - CMA-ES on small MLP, improved PID with learned gains
  - Simple behavioral cloning from PID trajectories
Phase 3: Iterate
  - Better controllers generate better data → retrain
  - Explore novel architectures

Track all results in src/eval/results.json. Always know current best score.
After each worker task, briefly note what you learned and what to do next.`;

const promptArg = process.argv
  .find((a) => a.startsWith("--prompt="))
  ?.split("=")
  .slice(1)
  .join("=");

const defaultPrompt = `Begin the research program for the comma.ai Controls Challenge.

Start with Phase 1:
1. Run the PID baseline locally (100 segments) to confirm scores
2. Study the tfpgh SOTA solution to understand what worked
3. Then move to Phase 2: design a compute-efficient controller that can beat PID`;

const prompt = promptArg || defaultPrompt;

async function main() {
  console.log(`\n=== rlclaw — comma controls challenge ===`);
  console.log(`Mode: orchestrator + single worker`);
  console.log(`GPU: RTX 5070 Ti (16GB VRAM)`);
  console.log(`Prompt: ${prompt.slice(0, 100)}...\n`);

  for await (const message of query({
    prompt,
    options: {
      cwd: process.cwd(),
      systemPrompt: ORCHESTRATOR_PROMPT,
      allowedTools: ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent"],
      agents: worker,
      maxTurns: 200,
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
    },
  })) {
    if ("result" in message) {
      console.log("\n=== Result ===");
      console.log(message.result);
    }
  }
}

main().catch(console.error);
