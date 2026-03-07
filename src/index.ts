import { query } from "@anthropic-ai/claude-agent-sdk";
import { agents } from "./agents/definitions";

// Allow running from within another Claude Code session
delete process.env.CLAUDECODE;

const SYSTEM_PROMPT = `You are the lead researcher orchestrating a team of agents to solve the comma.ai Controls Challenge.

GOAL: Find compute-efficient methods to minimize total_cost = (lataccel_cost * 50) + jerk_cost
for lateral car control. Beat the PID baseline (~107) and approach the SOTA (43.776) using
minimal GPU compute (3 Colab T4/A100 notebooks, 15 min per experiment).

Reference implementations:
  vendor/commaai/ — original challenge (PID baseline, tinyphysics simulator)
  vendor/tfpgh/  — best known solution (CMA-ES → PGTO → behavioral cloning, score 43.776)

Your team:
  arch-search     — explores controller architectures (small MLPs, SSMs, hybrid PID+NN)
  reward-optimizer — designs loss functions and training objectives
  data-engineer   — generates training data, manages pipelines
  evaluator       — runs benchmarks, tracks results, generates reports
  colab-manager   — manages the 3 Colab GPU notebooks and experiment execution

=== NOTEBOOK CHECKOUT SYSTEM ===
3 Colab GPU notebooks: notebook_01, notebook_02, notebook_03
Pool state tracked in: src/colab/pool_state.json
VS Code bridge API at: http://127.0.0.1:18808
Hard limit: 15 minutes per experiment.

When an agent needs GPU, they:
1. Check pool_state.json for an available notebook
2. Write experiment code into the notebook .ipynb file
3. POST to bridge /run to start it
4. Poll bridge /read-outputs to check completion
5. Update pool_state.json to release the notebook

=== RESEARCH STRATEGY ===
Phase 1: Understand the problem
  - Have evaluator run PID baseline to confirm scores
  - Have arch-search study the tfpgh solution architecture
  - Establish local eval pipeline (100 segments, fast CPU eval)

Phase 2: Quick wins
  - CMA-ES on a small MLP (can run in <15 min on GPU)
  - Improved PID with learned gains
  - Simple behavioral cloning from PID trajectories (no expensive PGTO)

Phase 3: Iterate
  - Train better controllers using data from Phase 2
  - Explore novel architectures that are compute-efficient
  - Self-improvement loop: best controller generates better data → retrain

Track all results in src/eval/results.json. Always know current best score.`;

const promptArg = process.argv
  .find((a) => a.startsWith("--prompt="))
  ?.split("=")
  .slice(1)
  .join("=");

const defaultPrompt = `Begin the research program for the comma.ai Controls Challenge.

First, set up the project:
1. Have the evaluator run the PID baseline locally (100 segments) to establish baseline scores
2. Have arch-search study vendor/tfpgh/ to understand the winning approach
3. Have data-engineer set up the data pipeline (download dataset if needed)

Then start Phase 2: design a compute-efficient controller that can be trained in under 15 minutes
on a single Colab GPU. Start with the simplest approach that could beat PID.`;

const prompt = promptArg || defaultPrompt;

async function main() {
  console.log(`\n=== rlclaw — comma controls challenge ===`);
  console.log(`Agents: ${Object.keys(agents).join(", ")}`);
  console.log(`Notebooks: 01, 02, 03 (15 min max each)`);
  console.log(`Prompt: ${prompt.slice(0, 100)}...\n`);

  for await (const message of query({
    prompt,
    options: {
      cwd: process.cwd(),
      systemPrompt: SYSTEM_PROMPT,
      allowedTools: ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent"],
      agents,
      maxTurns: 100,
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
