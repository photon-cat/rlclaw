import { query } from "@anthropic-ai/claude-agent-sdk";
import type { AgentDefinition } from "@anthropic-ai/claude-agent-sdk";
import * as fs from "fs";
import * as path from "path";

// Load .env before anything else
const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, "utf-8").split("\n")) {
    const match = line.match(/^([^#=]+)=(.*)$/);
    if (match) process.env[match[1].trim()] = match[2].trim();
  }
}

import { notify } from "./notify";
import {
  initTelemetry,
  recordTurn,
  recordLog,
  setStatus,
} from "./telemetry";

// Allow running from within another Claude Code session
delete process.env.CLAUDECODE;

const COMMANDS_FILE = path.join(__dirname, "..", "commands.txt");
const INPUT_FILE = path.join(__dirname, "discord_input.txt");

const CHALLENGE_CONTEXT = `=== COMMA CONTROLS CHALLENGE v2 ===
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

Scores: PID baseline ~85 (100 segs) | SOTA tfpgh v2 17.789 (MPC) | tfpgh v1 43.776 (BC)
  tfpgh v1: CMA-ES MLP (~55) -> GPU trajectory optimization (~43.2) -> behavioral cloning (43.776)
  tfpgh v2: MPC with NumPy inverse CDF sampling of physics model probabilities (17.789)

Reference code: vendor/commaai/ (v2 challenge), vendor/tfpgh/ (v2 SOTA solution)

=== WORKSPACE ===
All your work goes in workspace/ (gitignored). Do NOT write to src/.
  workspace/controllers/  — your controller implementations
  workspace/algos/        — training scripts
  workspace/checkpoints/  — saved models
  workspace/eval/         — evaluation results and logs
  workspace/results.json  — experiment result tracker

GPU: Local RTX 5070 Ti (16GB VRAM). Run experiments directly as python scripts.
Max 15 min per experiment. Controllers must run at 10Hz+, target <100K params.

Quick eval (~7s): cd vendor/commaai && python3 tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid`;

const worker: Record<string, AgentDefinition> = {
  worker: {
    description:
      "Executes research tasks: reads code, writes controllers, runs training, evaluates results. Use this for any task that requires tools.",
    prompt: `You are a research engineer working on the comma.ai Controls Challenge v2.
You receive specific tasks from the lead researcher and execute them thoroughly.

${CHALLENGE_CONTEXT}

You have access to all tools: read/write/edit files, run bash commands, search code.
Execute the task you're given completely, then report back with concrete results and metrics.
Be thorough but efficient. Always report numbers, not just "it worked".

IMPORTANT: If you are running a long task (training, large eval), periodically check commands.txt
in the project root. If it has content, read it, acknowledge by writing a short response to
src/discord_response.txt, and adapt your current work if the command is relevant.

All your work goes in workspace/ — controllers, training scripts, checkpoints, results.
Do NOT write to src/. Read vendor/ for reference but don't modify it.`,
    tools: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
  },
};

const ORCHESTRATOR_PROMPT = `You are the lead researcher for the comma.ai Controls Challenge v2.

${CHALLENGE_CONTEXT}

=== HOW YOU WORK ===
You have ONE worker agent ("worker") that executes tasks for you.
You PLAN what to do, then delegate ONE task at a time to the worker using the Agent tool.
When the worker returns, you REVIEW results, UPDATE your plan, and delegate the next task.

Each time you call the worker, give it a SPECIFIC, ACTIONABLE task. Not vague goals.
Good: "Read vendor/tfpgh/controllers/lookup.py and vendor/tfpgh/tinyphysics.py, summarize how the MPC lookup table was generated and how the physics model probabilities are sampled"
Bad: "Study the SOTA solution"

Good: "Write a CMA-ES training script to workspace/algos/cmaes_mlp.py that evolves a 2-hidden-layer MLP (32,16) to minimize total_cost on 100 segments. Run it for 5 minutes."
Bad: "Try to beat PID"

=== COMMANDS FILE ===
The user can steer you by writing to commands.txt in the project root (or @mentioning the bot in Discord).
Before each new worker task, check if commands.txt exists and has content.
If it does, read it, incorporate the instructions, then clear the file by writing an empty string.
This lets the user redirect your research without stopping the session.

IMPORTANT: After reading and processing a command from commands.txt, write a SHORT acknowledgment
to the file src/discord_response.txt so the user gets a reply in Discord. Keep it under 500 chars.
Example: "Got it, switching to MPC approach. Current task: studying tfpgh lookup.py"

=== ASKING FOR INPUT ===
If you hit a decision point where user input would be valuable (e.g., which direction to explore,
whether to spend GPU time on something risky, or you're stuck), write a message to the file
src/discord_input.txt with your question. The system will ping the user on Discord.
Then check commands.txt on subsequent turns for their response.
Only do this for genuine decision points — don't block on every step.

=== RESEARCH STRATEGY ===
Phase 1: Understand
  - Run PID baseline, study tfpgh v2 MPC solution
  - Understand how lookup.py and the inverse CDF sampling works
  - Establish fast local eval pipeline
Phase 2: Quick wins
  - Implement real-time MPC using physics model
  - CMA-ES on small MLP, improved PID with learned gains
Phase 3: Iterate
  - Better controllers generate better data -> retrain
  - Explore novel architectures and MPC variants

Track all results in workspace/results.json. Always know current best score.
After each worker task, briefly note what you learned and what to do next.`;

const promptArg = process.argv
  .find((a) => a.startsWith("--prompt="))
  ?.split("=")
  .slice(1)
  .join("=");

const defaultPrompt = `Begin the research program for the comma.ai Controls Challenge v2.

Start with Phase 1:
1. Run the PID baseline locally (100 segments) to confirm scores
2. Study the tfpgh v2 SOTA solution — especially how the MPC and inverse CDF sampling works
3. Then move to Phase 2: implement a real-time controller that can beat PID`;

const prompt = promptArg || defaultPrompt;

async function main() {
  console.log(`\n=== rlclaw — comma controls challenge v2 ===`);
  console.log(`Mode: orchestrator + single worker`);
  console.log(`GPU: RTX 5070 Ti (16GB VRAM)`);
  console.log(`Discord: notifications enabled`);
  console.log(`Dashboard: http://localhost:3000`);
  console.log(`Commands: write to commands.txt to steer research`);
  console.log(`Prompt: ${prompt.slice(0, 100)}...\n`);

  initTelemetry();
  await notify("Session started. Prompt: " + prompt.slice(0, 200));

  let turnCount = 0;

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
    // Check if orchestrator is asking for user input
    if (fs.existsSync(INPUT_FILE)) {
      const question = fs.readFileSync(INPUT_FILE, "utf-8").trim();
      if (question) {
        await notify(question + "\n\n_Reply by writing to `commands.txt` on the server._", "input");
        fs.unlinkSync(INPUT_FILE);
      }
    }

    if ("result" in message) {
      console.log("\n=== Result ===");
      console.log(message.result);
      recordLog("SESSION COMPLETE: " + message.result.slice(0, 300));
      setStatus("complete");
      await notify(message.result.slice(0, 500), "success");
    } else if ("message" in message) {
      turnCount++;
      const msg = message.message as any;

      // Track token usage
      if (msg?.usage) {
        recordTurn(msg);
      }

      if (msg?.content) {
        const text = Array.isArray(msg.content)
          ? msg.content
              .filter((b: any) => b.type === "text")
              .map((b: any) => b.text)
              .join("\n")
          : String(msg.content);
        if (text && msg.role === "assistant") {
          console.log(`\n[turn ${turnCount}] ${text.slice(0, 200)}`);
          recordLog(text.slice(0, 300));
          if (turnCount % 3 === 0 || text.includes("best score") || text.includes("Phase")) {
            await notify(text.slice(0, 500));
          }
        }
      }
    }
  }

  setStatus("complete");
  await notify("Session complete.", "success");
}

main().catch(async (err) => {
  console.error(err);
  setStatus("error");
  recordLog("ERROR: " + String(err).slice(0, 300));
  await notify(`Error: ${String(err).slice(0, 300)}`, "error");
  process.exit(1);
});
