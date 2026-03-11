import * as fs from "fs";
import * as path from "path";

const SESSION_DIR = path.join(__dirname, "..", "workspace", "sessions");
const CURRENT_SESSION = path.join(SESSION_DIR, "current.jsonl");
const SESSION_SUMMARY = path.join(SESSION_DIR, "summary.md");

export interface SessionEntry {
  time: string;
  role: "orchestrator" | "worker_dispatch" | "worker_result" | "command" | "system";
  content: string;
  agent?: string;
}

export function initSession(): void {
  fs.mkdirSync(SESSION_DIR, { recursive: true });

  // Archive previous session if it exists
  if (fs.existsSync(CURRENT_SESSION)) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const archive = path.join(SESSION_DIR, `session_${timestamp}.jsonl`);
    fs.renameSync(CURRENT_SESSION, archive);

    // Keep only last 10 archived sessions
    const archives = fs.readdirSync(SESSION_DIR)
      .filter(f => f.startsWith("session_") && f.endsWith(".jsonl"))
      .sort()
      .reverse();
    for (const old of archives.slice(10)) {
      fs.unlinkSync(path.join(SESSION_DIR, old));
    }
  }
}

export function appendSession(entry: SessionEntry): void {
  try {
    fs.appendFileSync(CURRENT_SESSION, JSON.stringify(entry) + "\n");
  } catch {}
}

export function writeSessionSummary(summary: string): void {
  try {
    fs.writeFileSync(SESSION_SUMMARY, summary);
  } catch {}
}

/** Build a resume prompt from the previous session + workspace state */
export function getResumeContext(): string | null {
  // Check for summary first
  if (fs.existsSync(SESSION_SUMMARY)) {
    const summary = fs.readFileSync(SESSION_SUMMARY, "utf-8").trim();
    if (summary) return summary;
  }

  // Fall back to reconstructing from session log
  const archives = fs.readdirSync(SESSION_DIR)
    .filter(f => f.startsWith("session_") && f.endsWith(".jsonl"))
    .sort()
    .reverse();

  if (archives.length === 0) return null;

  const lastSession = path.join(SESSION_DIR, archives[0]);
  const lines = fs.readFileSync(lastSession, "utf-8").trim().split("\n");
  if (lines.length === 0) return null;

  // Extract key events from last session
  const entries: SessionEntry[] = lines
    .map(l => { try { return JSON.parse(l); } catch { return null; } })
    .filter(Boolean);

  if (entries.length === 0) return null;

  // Build context from orchestrator messages (last 20)
  const orchestratorMsgs = entries
    .filter(e => e.role === "orchestrator")
    .slice(-20)
    .map(e => `- ${e.content.slice(0, 300)}`)
    .join("\n");

  const commands = entries
    .filter(e => e.role === "command")
    .slice(-5)
    .map(e => `- ${e.content.slice(0, 200)}`)
    .join("\n");

  return `=== PREVIOUS SESSION CONTEXT ===
The previous session crashed or was stopped. Here's what happened:

Orchestrator notes:
${orchestratorMsgs}

${commands ? `User commands received:\n${commands}\n` : ""}
Resume from where you left off. Check workspace/results.json for experiment results
and workspace/ for any work in progress. Do NOT repeat completed experiments.`;
}

/** Load results.json to give orchestrator current state */
export function getWorkspaceState(): string {
  const resultsPath = path.join(__dirname, "..", "workspace", "results.json");
  try {
    if (fs.existsSync(resultsPath)) {
      const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
      const exps = results.experiments || [];
      if (exps.length === 0) return "";

      const lines = exps.map((e: any) =>
        `  ${e.controller || e.tag || "unknown"}: total_cost=${e.total_cost}`
      ).join("\n");
      return `\n=== CURRENT EXPERIMENT RESULTS ===\n${lines}\n`;
    }
  } catch {}
  return "";
}
