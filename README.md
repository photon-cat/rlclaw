# rlclaw

Autonomous AI agent that optimizes low-level controllers for real-world systems. Give it a control problem, a simulator, and a GPU — it researches, implements, trains, and iterates 24/7 until it finds something good.

## First target: [comma.ai Controls Challenge](https://github.com/commaai/controls_challenge)

Lateral acceleration control for real cars. The agent runs on a single machine with an RTX 5070 Ti, autonomously writing code, training models, and evaluating results. A human steers high-level direction via Discord; the agent handles everything else.

### Results

Over ~24 hours of autonomous research, the agent:

- Ran **44 tracked experiments** across 19 controller architectures
- Wrote **107 optimization scripts** (CMA-ES, MPC, CEM, gradient-based, trajectory optimization)
- Achieved **13.89 total cost** on 100 segments (local eval, not yet submitted)

#### Where that sits on the [leaderboard](https://comma.ai/leaderboard)

The official leaderboard evaluates on the full dataset (scores differ from 100-segment local eval). Top entries worth studying:

| Rank | Who | Score | Method | Notes |
|---|---|---|---|---|
| 1 | haraschax | 35.97 | MPC + much compute | Leaderboard SOTA. Brute-force MPC, likely similar to our approach |
| 2 | tfpgh | 43.78 | PGTO + BC distillation | CMA-ES trajectory optimization, then distill into a neural net |
| 3 | utkarshgill | 44.83 | PPO + MPC | RL-trained policy combined with model-predictive control |
| 4 | bheijden | 45.76 | PPO | Pure reinforcement learning |
| 5 | ellenjxu | 48.08 | Feedback controller + evolution | Evolved custom controller |
| 6 | TheConverseEngineer | 48.47 | Tube MPC + 2-DOF PID | Classical robust control |
| 8 | dumanah | 49.26 | MPC w/ diffusion-style annealing | Interesting optimization approach |
| — | PID baseline | 110.25 | PID | Official baseline (full dataset) |

**Interesting patterns from the leaderboard:**
- Top 2 both use heavy offline compute (MPC/PGTO) — same direction our agent independently discovered
- PPO shows up twice in top 5 — worth exploring as a generalizable approach vs our precompute strategy
- The gap between #1 (35.97) and #2 (43.78) is huge — haraschax likely found a similar "exploit the deterministic sim" insight
- Most entries below rank 10 are PID variants with feedforward, scoring 50-80
- tfpgh's v2 approach (MPC with inverse CDF sampling, score 17.79) was shared publicly but isn't on the leaderboard — it may use a newer version of the challenge

#### Our approach vs the field

Our 13.89 on 100 segments is promising but comes with caveats:
- **Not yet submitted** to the official leaderboard (full-dataset eval will likely differ)
- **Precomputed lookup table**, not a generalizable controller — works by exploiting simulator determinism to find optimal per-segment action sequences
- The hash-based segment identification in the runtime controller is currently broken (the verification script confirms the score, but the deployable controller doesn't reproduce it)

The agent's research trajectory mirrors what the best human competitors found: MPC-based approaches with heavy offline compute dominate this challenge.

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
| Anything else | Routed to mediator (Opus) which can answer or steer research |

## License

MIT
