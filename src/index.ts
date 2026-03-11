import { query } from "@anthropic-ai/claude-agent-sdk";
import type { Query } from "@anthropic-ai/claude-agent-sdk";
import * as fs from "fs";
import * as path from "path";
import { spawn } from "child_process";

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
import {
  initSession,
  appendSession,
  writeSessionSummary,
  getResumeContext,
  getWorkspaceState,
} from "./session";

// Allow running from within another Claude Code session
delete process.env.CLAUDECODE;

const COMMANDS_FILE = path.join(__dirname, "..", "commands.txt");
const WORKER_DIR = path.join(__dirname, "..", "workspace", "workers");
const PROJECT_ROOT = path.join(__dirname, "..");

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
  workspace/workers/      — worker task/result files

GPU: Local RTX 5070 Ti (16GB VRAM). Run experiments directly as python scripts.
Controllers must run at 10Hz+ (real-time). Target <100K params for efficiency.

Quick eval (~7s): cd vendor/commaai && python3 tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid`;

const WORKER_SYSTEM = `You are a research engineer working on the comma.ai Controls Challenge v2.
You receive a specific task and execute it thoroughly.

${CHALLENGE_CONTEXT}

=== RULES ===
- Execute the task completely, then your output is your final report.
- Be thorough but efficient. Always report concrete numbers and results.
- All work goes in workspace/. Do NOT write to src/. Read vendor/ for reference only.
- All training scripts MUST include: periodic checkpointing to workspace/checkpoints/,
  metric logging to workspace/eval/, and workspace/results.json updates.`;

const numWorkers = parseInt(
  process.argv.find((a) => a.startsWith("--workers="))?.split("=")[1] ||
    process.env.RLCLAW_WORKERS ||
    "1",
  10
);

const ORCHESTRATOR_PROMPT = `You are the lead researcher for the comma.ai Controls Challenge v2.
You work autonomously. A mediator agent handles all user communication — you just focus on research.

!!! ABSOLUTE RULE — READ THIS FIRST !!!
You MUST background ALL python scripts. NEVER run python directly — it blocks the system.
ALWAYS use:
  nohup python3 script.py > workspace/eval/name.log 2>&1 & echo "PID=$!"
Then check with: tail -20 workspace/eval/name.log
The ONLY exception is one-liners under 10 seconds (e.g., python3 -c "print(1+1)").

${CHALLENGE_CONTEXT}

=== HOW YOU WORK ===
You have tools to read/write/edit files, run commands, and search code.

To DISPATCH A WORKER for independent tasks, write a file to workspace/workers/task_<name>.txt
with the task description. The system launches a separate Claude Code instance for each task.
Workers write results to workspace/workers/result_<name>.txt when done.
The system injects worker results into your conversation automatically.

Max ${numWorkers} concurrent workers. Workers are best for: code analysis, writing scripts, quick evals.
Track background PIDs in workspace/eval/running_pids.txt. Check nvidia-smi before GPU jobs.

=== INJECTED MESSAGES ===
You may receive injected messages from the mediator (user requests) or the system.
When you see "[Mediator directive]:", treat it as a high-priority research directive from the user.
Execute it, then continue your work. No need to write response files — the mediator handles Discord.
When you see "[System: ... Resume...]", continue autonomous research.

=== RESEARCH STRATEGY ===
Phase 1: Understand — run PID baseline, study tfpgh v2 MPC, establish eval pipeline
Phase 2: Quick wins — MPC, CMA-ES MLP, improved PID with learned gains
Phase 3: Iterate — better controllers -> better data -> retrain, novel architectures

Track all results in workspace/results.json. Always know current best score.`;

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

// ===== Worker Management =====

interface WorkerState {
  name: string;
  pid: number;
  startTime: number;
  taskFile: string;
  resultFile: string;
}

const activeWorkerProcs: Map<string, WorkerState> = new Map();

function ensureWorkerDir() {
  fs.mkdirSync(WORKER_DIR, { recursive: true });
}

/** Launch workers for any new task files */
function checkForNewTasks() {
  ensureWorkerDir();
  const files = fs.readdirSync(WORKER_DIR).filter(f => f.startsWith("task_") && f.endsWith(".txt"));

  for (const file of files) {
    const name = file.replace("task_", "").replace(".txt", "");
    const resultFile = path.join(WORKER_DIR, `result_${name}.txt`);

    if (activeWorkerProcs.has(name)) continue;
    if (fs.existsSync(resultFile)) continue;
    if (activeWorkerProcs.size >= numWorkers) continue;

    const taskPath = path.join(WORKER_DIR, file);
    const task = fs.readFileSync(taskPath, "utf-8").trim();
    if (!task) continue;

    console.log(`[worker:${name}] Launching...`);
    recordLog(`-> worker:${name}: ${task.slice(0, 200)}`);
    appendSession({ time: new Date().toISOString(), role: "worker_dispatch", content: task.slice(0, 2000), agent: name });

    const workerPrompt = `${task}\n\nWhen done, report your complete results with specific numbers and file paths.`;

    const child = spawn("claude", [
      "--dangerously-skip-permissions",
      "-p", workerPrompt,
      "--output-file", resultFile,
    ], {
      cwd: PROJECT_ROOT,
      env: { ...process.env, CLAUDECODE: "" },
      stdio: "ignore",
      detached: true,
    });

    child.unref();

    if (child.pid) {
      activeWorkerProcs.set(name, {
        name,
        pid: child.pid,
        startTime: Date.now(),
        taskFile: taskPath,
        resultFile,
      });
    }
  }
}

/** Check for completed workers, return result messages to inject */
function checkWorkerResults(): string[] {
  const messages: string[] = [];

  for (const [name, worker] of activeWorkerProcs) {
    if (fs.existsSync(worker.resultFile)) {
      try {
        const result = fs.readFileSync(worker.resultFile, "utf-8").trim();
        if (result) {
          const elapsed = Math.round((Date.now() - worker.startTime) / 1000);
          console.log(`[worker:${name}] Completed in ${elapsed}s`);
          recordLog(`<- worker:${name}: ${result.slice(0, 200)}`);
          appendSession({ time: new Date().toISOString(), role: "worker_result", content: result.slice(0, 2000), agent: name });
          messages.push(`[Worker "${name}" completed in ${elapsed}s]\n${result}`);
          activeWorkerProcs.delete(name);
          try { fs.unlinkSync(worker.taskFile); } catch {}
        }
      } catch {}
    } else {
      // Check if process died
      try {
        process.kill(worker.pid, 0);
      } catch {
        const elapsed = Math.round((Date.now() - worker.startTime) / 1000);
        console.log(`[worker:${name}] Died after ${elapsed}s`);
        recordLog(`!! worker:${name} died`);
        appendSession({ time: new Date().toISOString(), role: "worker_result", content: "Worker died without results.", agent: name });
        messages.push(`[Worker "${name}" failed — died after ${elapsed}s without results]`);
        activeWorkerProcs.delete(name);
        try { fs.unlinkSync(worker.taskFile); } catch {}
      }
    }
  }

  return messages;
}

function getWorkerStatus(): string {
  if (activeWorkerProcs.size === 0) return "";
  const lines = [...activeWorkerProcs.values()].map(w => {
    const elapsed = Math.round((Date.now() - w.startTime) / 1000);
    return `  - ${w.name}: running ${elapsed}s (PID ${w.pid})`;
  });
  return `\nActive workers:\n${lines.join("\n")}`;
}

// ===== Main =====

async function main() {
  console.log(`\n=== rlclaw — comma controls challenge v2 ===`);
  console.log(`Mode: orchestrator + ${numWorkers} async worker(s)`);
  console.log(`GPU: RTX 5070 Ti (16GB VRAM)`);
  console.log(`Discord: notifications enabled`);
  console.log(`Dashboard: http://localhost:3000`);
  console.log(`Commands: @mention in Discord or write to commands.txt`);
  console.log(`Prompt: ${prompt.slice(0, 100)}...\n`);

  initTelemetry();
  initSession();
  ensureWorkerDir();

  const resumeContext = getResumeContext();
  const workspaceState = getWorkspaceState();
  let effectivePrompt = prompt;

  if (resumeContext) {
    console.log("Resuming from previous session...\n");
    effectivePrompt = `${resumeContext}\n${workspaceState}\n\nOriginal goal: ${prompt}\n\nContinue the research. Do NOT repeat already-completed experiments.`;
    await notify("Resuming from previous session");
  } else {
    await notify("Session started. Prompt: " + prompt.slice(0, 200));
  }

  // Start the orchestrator query
  const session = query({
    prompt: effectivePrompt,
    options: {
      cwd: PROJECT_ROOT,
      systemPrompt: ORCHESTRATOR_PROMPT,
      allowedTools: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
      maxTurns: 500,
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
    },
  });

  // Poll for mediator commands and worker results, inject into session
  let lastSeenCmd = "";

  const pollInterval = setInterval(async () => {
    try {
      // Check for commands from the mediator
      if (fs.existsSync(COMMANDS_FILE)) {
        const cmd = fs.readFileSync(COMMANDS_FILE, "utf-8").trim();
        if (cmd && cmd !== lastSeenCmd) {
          lastSeenCmd = cmd;
          fs.writeFileSync(COMMANDS_FILE, "");

          if (cmd === "!resume") {
            console.log(`[system] Resume triggered`);
            await session.interrupt();
            await session.send("[System: Resume autonomous research work.]");
          } else {
            // Mediator directive — interrupt current work and inject
            appendSession({ time: new Date().toISOString(), role: "command", content: cmd });
            console.log(`[mediator] Interrupting orchestrator for directive: ${cmd.slice(0, 100)}`);
            await session.interrupt();
            await session.send(`[Mediator directive]: ${cmd}`);
          }
        }
      }

      // Check for new worker tasks
      checkForNewTasks();

      // Check for completed worker results
      const workerResults = checkWorkerResults();
      for (const result of workerResults) {
        await session.send(result);
      }
    } catch (err) {
      // Ignore errors in poll loop (session may be between turns)
    }
  }, 3_000);

  // Stream orchestrator messages
  let turnCount = 0;
  let lastSummary = "";

  try {
    for await (const message of session) {
      if ("result" in message) {
        console.log("\n=== Result ===");
        console.log(message.result);
        recordLog("SESSION COMPLETE: " + message.result.slice(0, 300));
        setStatus("complete");
        await notify(message.result.slice(0, 500), "success");
      } else if ("message" in message) {
        turnCount++;
        const msg = message.message as any;

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
            appendSession({ time: new Date().toISOString(), role: "orchestrator", content: text });
            lastSummary = text;

            if (turnCount % 5 === 0) {
              writeSessionSummary(
                `Last updated: ${new Date().toISOString()}\nTurn: ${turnCount}\n\nLast orchestrator output:\n${lastSummary.slice(0, 1000)}\n${getWorkerStatus()}\n${getWorkspaceState()}`
              );
            }

            if (turnCount % 3 === 0 || text.includes("best score") || text.includes("Phase")) {
              await notify(text.slice(0, 500));
            }
          }
        }
      }
    }
  } finally {
    clearInterval(pollInterval);
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
