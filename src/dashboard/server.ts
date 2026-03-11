import * as http from "http";
import * as fs from "fs";
import * as path from "path";

const PORT = 3000;
const TELEMETRY_PATH = path.join(__dirname, "..", "telemetry.json");
const RESULTS_PATH = path.join(__dirname, "..", "eval", "results.json");
const COMMANDS_FILE = path.join(__dirname, "..", "..", "commands.txt");
const SESSION_LOG = path.join(__dirname, "..", "..", "workspace", "sessions", "current.jsonl");
const AGENT_LOG = "/tmp/rlclaw-session.log";
const STATIC_DIR = path.join(__dirname, "ui");

function readJson(p: string): unknown {
  try {
    return JSON.parse(fs.readFileSync(p, "utf-8"));
  } catch {
    return null;
  }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://localhost:${PORT}`);
  const pathname = url.pathname;

  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  if (pathname === "/api/telemetry") {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(readJson(TELEMETRY_PATH) || { status: "no session" }));
    return;
  }

  if (pathname === "/api/results") {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(readJson(RESULTS_PATH) || { experiments: [], baselines: {} }));
    return;
  }

  if (pathname === "/api/command" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      try {
        const { command } = JSON.parse(body);
        if (command) {
          fs.writeFileSync(COMMANDS_FILE, command);
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ ok: true }));
        } else {
          res.writeHead(400);
          res.end(JSON.stringify({ error: "empty command" }));
        }
      } catch {
        res.writeHead(400);
        res.end(JSON.stringify({ error: "invalid json" }));
      }
    });
    return;
  }

  if (pathname === "/api/session-log") {
    res.setHeader("Content-Type", "application/json");
    try {
      const lines = fs.readFileSync(SESSION_LOG, "utf-8").trim().split("\n");
      const entries = lines
        .map((l) => { try { return JSON.parse(l); } catch { return null; } })
        .filter(Boolean)
        .slice(-100);
      res.end(JSON.stringify(entries));
    } catch {
      res.end(JSON.stringify([]));
    }
    return;
  }

  if (pathname === "/api/agent-output") {
    res.setHeader("Content-Type", "application/json");
    const { execSync } = await import("child_process");
    try {
      const output = execSync(
        "journalctl -u rlclaw-agent --no-pager -n 200 --output=short-iso 2>/dev/null || tail -200 /tmp/rlclaw-session.log 2>/dev/null || echo 'no logs'",
        { timeout: 5000 }
      ).toString();
      res.end(JSON.stringify({ output }));
    } catch {
      res.end(JSON.stringify({ output: "" }));
    }
    return;
  }

  if (pathname === "/api/gpu") {
    const { execSync } = await import("child_process");
    try {
      const smi = execSync(
        "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
        { timeout: 5000 }
      ).toString().trim();
      const [name, temp, util, memUsed, memTotal] = smi.split(", ");
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ name, temp: +temp, util: +util, memUsedMB: +memUsed, memTotalMB: +memTotal }));
    } catch {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ error: "nvidia-smi failed" }));
    }
    return;
  }

  // Static files
  if (pathname === "/" || pathname === "/index.html") {
    try {
      const content = fs.readFileSync(path.join(STATIC_DIR, "index.html"), "utf-8");
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end(content);
    } catch {
      res.writeHead(404);
      res.end("Not found");
    }
    return;
  }

  res.writeHead(404);
  res.end("Not found");
});

server.listen(PORT, () => {
  console.log(`rlclaw dashboard: http://localhost:${PORT}`);
});
