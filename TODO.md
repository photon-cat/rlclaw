# TODO

## Agent file management

The agent writes all its work to `workspace/` (gitignored). Currently there's no good way to review what it did over time.

### What exists
- **Workspace tarball backups** every 30 min via cron (`backups/workspace_*.tar.gz`), 48 retained (24h rolling window)
- **Session logs** (`workspace/sessions/*.jsonl`) — truncated text of orchestrator output, 10 archived sessions
- **Telemetry snapshots** (`backups/telemetry_*.json`) — cost/usage stats every 30 min

### What's missing
- **No git history of workspace/** — it's gitignored so there's no diff-level view of changes. Can't see "the agent changed line 42 of mpc_multipass.py at 3am"
- **No file-level changelog** — session log captures what the agent *said*, not what it actually wrote. Tool calls (reads/writes/edits) aren't logged
- **Backups are opaque tarballs** — to see what changed between two points you'd extract two tarballs and diff manually. No tooling for this
- **Session logs are thin** — just `content.slice(0, 2000)` of text output. The actual Bash commands, file edits, and their results are lost

### Ideas
- [ ] Git-commit workspace/ periodically (separate repo or branch) so you get real version history with diffs
- [ ] Log tool calls (file writes, bash commands) to a structured log alongside the session JSONL
- [ ] Build a CLI to diff workspace between two backup timestamps (`rlclaw diff 2026-03-10T06 2026-03-10T12`)
- [ ] Tag backups with the current best score so you can find "the backup right before/after the score improved"
- [ ] Store the full Claude Code conversation transcript, not just truncated orchestrator text

## Command delivery / orchestrator responsiveness

The orchestrator can get stuck in long tool calls (sleep, tail, GPU monitoring loops) and won't see injected commands until the current turn finishes. Fixed: `session.interrupt()` now called before `session.send()` for mediator directives, which kills the running tool call and forces the orchestrator to read the message immediately.

Still TODO:
- [ ] Worker result injection could also block the poll loop — consider fire-and-forget or a queue
- [ ] Add a watchdog: if orchestrator hasn't logged a turn in N minutes, auto-interrupt and inject a "what are you doing?" prompt
- [ ] Consider sending worker results without interrupt (lower priority, shouldn't disrupt flow)

## Controls challenge

**Critical: sim mismatch bug.** The agent's reimplemented simulator (`rebuild_verified.py`, `gpu_refine_onnx.py`, etc.) uses `np.random.random()` + `searchsorted` for token sampling, while the official `tinyphysics.py` uses `np.random.choice(N, p=probs)`. These consume the RNG stream differently, causing trajectory divergence from step 101 onward. All "13.89" scores were computed against the wrong simulator. The precomputed action sequences don't transfer to the real sim.

Additionally, the entire precompute-per-segment approach (lookup tables keyed by segment hash) is a dead end for the challenge — it exploits deterministic seeding rather than building a real controller that generalizes to unseen data.

- [x] ~~Fix the hash mismatch in lookup_combined.py~~ Moot — the whole approach is invalid
- [ ] **Fix the sim reimplementation** to use `np.random.choice` matching the official sim, if precompute is still desired
- [ ] **Pivot to generalizable controllers** — the challenge is about stochastic control, not memorization
- [ ] Explore PPO — shows up twice in leaderboard top 5, would give a real policy
- [ ] Study haraschax's approach (#1, 35.97) — "MPC + much compute" but presumably a real controller
- [ ] Try distilling precomputed action sequences into a small neural net (like tfpgh's BC step)
- [ ] Submit to the official leaderboard (full dataset eval, not just 100 segments)
