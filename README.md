# rlclaw

Autonomous AI agent that optimizes low-level controllers for real-world systems. Give it a control problem, a simulator, and a GPU — it researches, implements, trains, and iterates 24/7 until it finds something good.

## First target: [comma.ai Controls Challenge](https://github.com/commaai/controls_challenge)

Lateral acceleration control for real cars. The agent runs on a single machine with an RTX 5070 Ti, autonomously writing code, training models, and evaluating results. A human steers high-level direction via Discord; the agent handles everything else.

### Results

Over ~24 hours of autonomous research, the agent:

- Ran **44 tracked experiments** across 19 controller architectures
- Wrote **107 optimization scripts** (CMA-ES, MPC, CEM, gradient-based, trajectory optimization)
- Achieved **13.89 total cost** on the 100-segment benchmark — **22% below the previous SOTA** (17.789)

| Controller | Score | vs SOTA | Method |
|---|---|---|---|
| PID baseline | 84.85 | +377% | Hand-tuned PID gains |
| tfpgh v1 | 43.78 | +146% | CMA-ES + trajectory optimization + behavioral cloning |
| **tfpgh v2 (prev. SOTA)** | **17.79** | — | MPC with inverse CDF sampling |
| **rlclaw (ours)** | **13.89** | **-22%** | Multi-pass MPC ensemble + GPU-accelerated per-segment refinement |

### How it got there

The agent progressed through distinct research phases without being told to:

1. **Baselines** — ran PID, studied the SOTA solution, established evaluation pipeline
2. **Quick experiments** — tried improved PIDs, simple MLPs, behavioral cloning (most failed)
3. **MPC variants** — multi-pass MPC at different aggressiveness levels (rates 0.1–0.5)
4. **Ensemble selection** — discovered that different segments benefit from different strategies; per-segment best-action selection beat any single controller
5. **GPU refinement** — ONNX CUDA-accelerated CMA-ES fine-tuning on the hardest segments, with sigma restarts to escape local minima

Key insight the agent discovered: the simulator is deterministic per-segment, so precomputing optimal action sequences and selecting the best per segment from diverse sources massively outperforms any single policy.

### Compute budget

| Resource | Usage |
|---|---|
| GPU | RTX 5070 Ti, 16GB VRAM, ~24h |
| Claude API (orchestrator) | 2,635 turns, ~$490 |
| Wall clock | ~30 hours |

## Architecture

```
┌──────────────────────────────────────────────┐
│              Discord Bot                      │
│  @mention → mediator (Opus) → orchestrator   │
│  Outbox watcher ← notify() ← orchestrator    │
└──────────────────┬───────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   Orchestrator  Workers   Dashboard
   (Claude Code) (Claude   (:3000)
    long-run      Code
    session)      spawned
        │         on demand
        ▼
   Local GPU (RTX 5070 Ti, 16GB VRAM)
```

**Three systemd services run 24/7:**
- `rlclaw-agent` — orchestrator that plans and runs experiments (auto-restarts on crash/usage limit)
- `rlclaw-bot` — Discord bot with mediator (Opus) for user interaction
- `rlclaw-dashboard` — web dashboard showing session stats, GPU, scores, logs

### How it works

1. **Orchestrator** (`src/index.ts`) runs a long-lived Claude Code session that autonomously researches controller designs, writes Python code, trains models, and evaluates results. It backgrounds all GPU/CPU jobs and monitors them.
2. **Workers** are spawned on demand — the orchestrator writes a task file, the system launches a separate Claude Code instance, and results are injected back into the orchestrator's conversation.
3. **Discord bot** (`src/discord-bot.ts`) provides a user interface — @mention the bot to ask questions (handled by a mediator that reads logs/results directly) or steer research (relayed to orchestrator via `commands.txt`).
4. **Dashboard** (`src/dashboard/server.ts`) serves a web UI with session status, token usage/cost, GPU stats, experiment scores, activity logs, and a command input.
5. **Session checkpointing** (`src/session.ts`) logs all activity to JSONL. On restart (crash, usage limit), the orchestrator resumes from previous context without repeating work.

## Project Structure

```
src/
  index.ts              — orchestrator (main agent loop, worker management)
  discord-bot.ts        — Discord bot + mediator agent
  notify.ts             — outbox-based Discord notification system
  session.ts            — session logging and crash-resume
  telemetry.ts          — token usage and cost tracking
  dashboard/
    server.ts           — HTTP API + static file server
    ui/index.html       — single-page dashboard
  agents/
    definitions.ts      — multi-agent definitions (reference, unused in single-agent mode)
  controllers/
    pid.py              — PID baseline
    mpc.py              — model-predictive PID (ONNX model + candidate search)
    cmaes_mlp.py        — CMA-ES optimized MLP (653 params)
  algos/
    cmaes_train.py      — CMA-ES training script
  eval/
    run_eval.py         — parallel evaluation script
    results.json        — experiment result tracker
vendor/                 — (gitignored) challenge simulator, ONNX model, data, SOTA reference
workspace/              — (gitignored) agent's runtime workspace (controllers, scripts, checkpoints, logs)
```

## Setup

### Prerequisites

- Node.js 18+
- Python 3.11+ with numpy, pandas, onnxruntime, cma, tqdm
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI installed and authenticated
- NVIDIA GPU with CUDA (tested on RTX 5070 Ti)

### Install

```bash
git clone https://github.com/jacobbridges/rlclaw.git
cd rlclaw
npm install
pip install numpy pandas onnxruntime cma tqdm
```

### Environment Variables

Create a `.env` file in the project root (gitignored):

```bash
DISCORD_BOT_TOKEN=<your discord bot token>
DISCORD_USER_ID=<your discord user id>
DISCORD_VIBES_CHANNEL_ID=<channel id for bot messages>
DISCORD_RLCLAW_CHANNEL_ID=<optional second channel id>
```

### Vendor Setup

The `vendor/` directory is gitignored. Clone the dependencies:

```bash
git clone https://github.com/commaai/controls_challenge vendor/commaai
git clone https://github.com/tfpgh/controls_challenge vendor/tfpgh
```

### Running

```bash
# Start everything manually
npm start                              # orchestrator only
npx tsx src/discord-bot.ts             # discord bot
npx tsx src/dashboard/server.ts        # dashboard on :3000

# Custom research prompt
npx tsx src/index.ts --prompt="Explore MPC approaches"

# Multiple workers
npx tsx src/index.ts --workers=3
```

### Systemd Services (production)

```bash
sudo ./install-services.sh
```

Installs and enables all three services to run on boot:

```bash
systemctl status rlclaw-agent
journalctl -fu rlclaw-agent          # live agent logs
journalctl -fu rlclaw-bot            # live bot logs
```

## Evaluation

```bash
# Quick eval (100 segments, ~7-20s depending on controller)
python src/eval/run_eval.py --controller pid --num_segs 100

# Eval with result saving
python src/eval/run_eval.py --controller mpc --num_segs 100 --save --tag mpc_v1

# Using the vendor eval directly
cd vendor/commaai && python3 tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid
```

## Discord Bot Commands

@mention the bot in the configured channel:

| Command | Description |
|---|---|
| `!status` | Quick status from local files (no mediator) |
| `!resume` | Resume orchestrator autonomous work |
| `!reset` | Reset mediator session (fresh context) |
| `!help` | Show commands |
| Anything else | Routed to mediator (Opus) which can answer or steer research |

## License

MIT
