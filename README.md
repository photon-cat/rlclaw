# rlclaw

Autonomous AI research agent for the [comma.ai Controls Challenge v2](https://github.com/commaai/controls_challenge). A Claude Code orchestrator runs 24/7, designing, training, and evaluating controllers for lateral car steering — with a Discord bot for monitoring and steering the research.

## Goal

Minimize `total_cost = (lataccel_cost * 50) + jerk_cost` for lateral acceleration control. Beat PID baseline (~81), approach SOTA (17.789).

| Benchmark | Score | Method |
|---|---|---|
| PID baseline | ~81 | Proportional-integral-derivative |
| tfpgh v1 | 43.776 | CMA-ES + trajectory optimization + behavioral cloning |
| tfpgh v2 (SOTA) | 17.789 | MPC with inverse CDF sampling of physics model |

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

**Three systemd services:**
- `rlclaw-agent` — orchestrator that plans and runs experiments (auto-restarts on crash/usage limit)
- `rlclaw-bot` — Discord bot with mediator (Opus) for user interaction
- `rlclaw-dashboard` — web dashboard showing session stats, GPU, scores, logs

### How it works

1. **Orchestrator** (`src/index.ts`) runs a long-lived Claude Code session that autonomously researches controller designs, writes code, trains models, and evaluates results.
2. **Workers** are spawned on demand — the orchestrator writes a task file to `workspace/workers/task_<name>.txt`, the system launches a separate Claude Code instance, and results are injected back.
3. **Discord bot** (`src/discord-bot.ts`) provides a user interface — @mention the bot to ask questions (handled by a mediator agent that reads logs/results directly) or steer research (relayed to orchestrator via `commands.txt`).
4. **Dashboard** (`src/dashboard/server.ts`) serves a web UI showing session status, token usage/cost, GPU stats, experiment scores, activity logs, and a command input.
5. **Notifications** (`src/notify.ts`) writes messages to an outbox directory; the Discord bot picks them up and sends to the channel.
6. **Session checkpointing** (`src/session.ts`) logs all orchestrator/worker activity to JSONL. On restart, the orchestrator resumes from the previous session context.

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
    definitions.ts      — multi-agent definitions (reference, currently unused)
  controllers/
    __init__.py          — BaseController import
    pid.py              — PID baseline (P=0.195, I=0.1, D=-0.053)
    mpc.py              — Model-predictive PID (ONNX model + candidate search)
    cmaes_mlp.py        — CMA-ES optimized MLP (653 params, 2-hidden-layer)
  algos/
    cmaes_train.py      — CMA-ES training script for the MLP controller
  eval/
    run_eval.py         — parallel evaluation script
    results.json        — experiment result tracker
vendor/                 — (gitignored) vendored dependencies
  commaai/              — controls challenge simulator, ONNX model, data
  tfpgh/                — SOTA reference solution
workspace/              — (gitignored) runtime workspace for agent output
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
# Controls challenge simulator + data
git clone https://github.com/commaai/controls_challenge vendor/commaai

# SOTA reference solution
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

Install all three services to run on boot:

```bash
sudo ./install-services.sh
```

This installs and enables:
- `rlclaw-agent` — restarts every 10 min on exit (handles Claude Code usage limits)
- `rlclaw-bot` — always-on Discord bot
- `rlclaw-dashboard` — always-on web dashboard

```bash
# Manage services
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

## Controllers

| Controller | File | Params | Description |
|---|---|---|---|
| PID | `src/controllers/pid.py` | 3 | Classic PID with tuned gains |
| MPC | `src/controllers/mpc.py` | 3 (PID) + model | PID + ONNX model-based candidate search |
| CMA-ES MLP | `src/controllers/cmaes_mlp.py` | 653 | 2-hidden-layer MLP optimized with CMA-ES |

### Training

```bash
# CMA-ES training (20 segments, 5 min)
cd /home/jacob/rlclaw && python3 -m src.algos.cmaes_train --num_segs 20 --max_time 300

# Resume from checkpoint
python3 -m src.algos.cmaes_train --num_segs 20 --max_time 300 --resume
```

## Discord Bot Commands

@mention the bot in the configured channel:

| Command | Description |
|---|---|
| `!status` | Quick status from local files (no mediator) |
| `!resume` | Resume orchestrator autonomous work |
| `!reset` | Reset mediator session (fresh context) |
| `!help` | Show commands |
| Anything else | Routed to mediator (Opus) which can answer questions or steer research |

## Colab CLI

A standalone CLI tool for managing Google Colab GPU runtimes and executing Python code from the terminal. Located in `colab-cli/`. No browser or VS Code needed after initial auth.

### Setup

1. Install the [Google Colab VS Code extension](https://marketplace.visualstudio.com/items?itemName=google.colab) (just needs to be installed, not running)

2. Extract OAuth credentials from the extension:
```bash
# Client ID
grep -oE '[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com' \
  ~/.vscode/extensions/google.colab-*/out/extension.js | head -1

# Client Secret
grep -oE 'GOCSPX-[A-Za-z0-9_-]+' \
  ~/.vscode/extensions/google.colab-*/out/extension.js | head -1
```

3. Create `colab-cli/.env`:
```bash
COLAB_CLIENT_ID=<client-id-from-above>
COLAB_CLIENT_SECRET=<secret-from-above>
```

4. Install and authenticate:
```bash
cd colab-cli && npm install
npx colab auth    # opens browser, one-time
```

### Quick Start

```bash
cd colab-cli

# Check your account
npx colab info

# Create a GPU runtime
npx colab create GPU T4      # or A100, H100, L4

# Run code on it
npx colab exec 'import torch; print(torch.cuda.get_device_name(0))'

# Run a file
npx colab exec -f train.py

# Interactive REPL
npx colab shell

# List active runtimes
npx colab ls
```

See `colab-cli/README.md` for full documentation.

## License

MIT
