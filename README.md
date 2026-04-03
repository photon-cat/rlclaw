# rlclaw

A general-purpose agent environment for reinforcement learning research. Claude Code agents get compute, tools, and autonomy to run experiments, iterate on ideas, and report novel findings.

## How It Works

You define a **research problem**. rlclaw spins up a team of specialist Claude Code agents that autonomously:

- Study reference implementations and papers
- Design and implement approaches
- Run GPU experiments locally (15 min max each)
- Evaluate results, track metrics, iterate
- Report back with findings

Each problem is fully isolated — its own agents, results, and workspace. Multiple problems can run concurrently.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Dashboard (:3000)                 │
│  Problem status • Experiment logs                   │
│  Results comparison • Agent activity • GPU usage    │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │Problem 1│   │Problem 2│   │Problem 3│
   │ comma   │   │ (next)  │   │  ...    │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │  Agent  │   │  Agent  │   │  Agent  │
   │  Team   │   │  Team   │   │  Team   │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              ┌─────────────────┐
              │  Local GPU      │
              │  RTX 5070 Ti    │
              └─────────────────┘
```

### Agent Teams

Each problem gets its own orchestrator and specialist agents. For the comma controls challenge:

| Agent | Role |
|---|---|
| **orchestrator** | Breaks down the problem, delegates, tracks progress |
| **arch-search** | Explores controller architectures |
| **reward-optimizer** | Designs loss functions and training objectives |
| **data-engineer** | Builds data pipelines and generates training data |
| **evaluator** | Runs benchmarks, tracks results |

## Setup

### Prerequisites

- Node.js 18+
- Python 3.11+ with Jupyter (`pip install jupyter nbclient`)
- [Claude Code](https://claude.ai/claude-code) with Max subscription
- Local GPU (RTX 5070 Ti or similar)

### Install

```bash
git clone https://github.com/your-org/rlclaw.git
cd rlclaw
npm install
```

## Usage

### Run a research problem

```bash
# Start the comma controls challenge
npm start

# Custom research prompt
npx tsx src/index.ts --prompt="Explore whether a tiny transformer (< 50K params) can beat PID for lateral control"
```

### Adding a new problem

Create a new problem directory under `src/problems/`:

```
src/problems/my-problem/
  index.ts          — orchestrator with problem-specific system prompt
  agents.ts         — specialist agent definitions
  eval/             — evaluation code and results
  controllers/      — implementations
```

Each problem is self-contained. The orchestrator imports from shared infra but has its own agents, prompts, and workspace.

## Current Problems

### 1. comma Controls Challenge

**Goal:** Minimize `total_cost = (lataccel_cost * 50) + jerk_cost` for lateral car steering control.

| Benchmark | Score | Notes |
|---|---|---|
| PID baseline | ~73 | Simple proportional-integral-derivative |
| SOTA (tfpgh) | 43.776 | CMA-ES -> GPU trajectory optimization -> behavioral cloning |
| Our target | < 60 | Compute-efficient, trainable in 15 min on a single GPU |

Reference code in `vendor/commaai/` and `vendor/tfpgh/`.

## Project Structure

```
rlclaw/
├── src/
│   ├── index.ts                — main entry point
│   ├── agents/
│   │   └── definitions.ts      — agent team definitions
│   ├── controllers/            — controller implementations
│   ├── algos/                  — training scripts and configs
│   └── eval/
│       └── results.json        — experiment result tracker
├── vendor/
│   ├── commaai/                — controls challenge + dataset
│   └── tfpgh/                  — SOTA reference solution
├── CLAUDE.md                   — agent instructions
├── package.json
└── tsconfig.json
```

## How It's Built

- **Agent SDK** (`@anthropic-ai/claude-agent-sdk`) — spawns Claude Code agents with tool access. Authenticated via Max subscription, no API key needed.
- **Local GPU** — RTX 5070 Ti (16GB VRAM) for training and evaluation.

The agents have full access to `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, and can spawn sub-agents via `Agent`. They operate on the local filesystem, run Python scripts, and trigger GPU experiments directly.

## License

MIT
