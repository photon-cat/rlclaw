import { Client, GatewayIntentBits, Events, TextChannel, Message } from "discord.js";
import {
  unstable_v2_createSession,
  unstable_v2_resumeSession,
} from "@anthropic-ai/claude-agent-sdk";
import type { SDKSession, SDKMessage, SDKSessionOptions } from "@anthropic-ai/claude-agent-sdk";
import * as fs from "fs";
import * as path from "path";

// Load .env
const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, "utf-8").split("\n")) {
    const match = line.match(/^([^#=]+)=(.*)$/);
    if (match) process.env[match[1].trim()] = match[2].trim();
  }
}

const TOKEN = process.env.DISCORD_BOT_TOKEN;
const VIBES_CHANNEL_ID = process.env.DISCORD_VIBES_CHANNEL_ID;
const RLCLAW_CHANNEL_ID = process.env.DISCORD_RLCLAW_CHANNEL_ID;
const USER_ID = process.env.DISCORD_USER_ID;
const COMMANDS_FILE = path.join(__dirname, "..", "commands.txt");
const OUTBOX_DIR = path.join(__dirname, "discord_outbox");
const TELEMETRY_PATH = path.join(__dirname, "telemetry.json");
const SESSION_LOG = path.join(__dirname, "..", "workspace", "sessions", "current.jsonl");
const RESULTS_PATH = path.join(__dirname, "..", "workspace", "results.json");
const PROJECT_ROOT = path.join(__dirname, "..");

if (!TOKEN || !VIBES_CHANNEL_ID) {
  console.error("Missing DISCORD_BOT_TOKEN or DISCORD_VIBES_CHANNEL_ID in .env");
  process.exit(1);
}

// ===== Mediator =====

const MEDIATOR_SYSTEM = `You are the user-facing interface for rlclaw, an autonomous research agent working on the comma.ai Controls Challenge v2.

You sit between the user (who talks to you via Discord) and the orchestrator (a long-running Claude Code instance doing research). Your job:

1. ANSWER STATUS QUESTIONS DIRECTLY — read telemetry, results, session logs, eval logs yourself. Don't bother the orchestrator for "what's the score?" or "what's it doing?"

2. RELAY RESEARCH DIRECTIVES — when the user wants to change direction, give new tasks, or steer research, write a clear command to the commands file. Be specific — translate casual user language into precise orchestrator instructions.

3. CONTROL RUNNING PROCESSES — you can kill PIDs, check nvidia-smi, tail logs, etc. If the user says "stop that" or "try something else", you can act directly.

4. BE CONCISE — your responses go to Discord (2000 char limit). Keep replies short and informative. No fluff.

=== WORKSPACE LAYOUT ===
workspace/results.json — experiment results tracker
workspace/eval/*.log — evaluation output logs
workspace/controllers/ — controller implementations
workspace/algos/ — training/optimization scripts
workspace/checkpoints/ — saved models
workspace/sessions/current.jsonl — orchestrator session log (JSONL, roles: orchestrator/command/worker_*)
src/telemetry.json — turns, cost, usage stats

=== ORCHESTRATOR COMMUNICATION ===
To send a directive to the orchestrator, use Bash:
  echo "your directive here" > ${COMMANDS_FILE}

The orchestrator polls this file every 3 seconds. It will see "[Mediator directive]: <your text>".

To check what the orchestrator is doing, read the session log:
  Read last entries from: ${SESSION_LOG}

=== CHALLENGE CONTEXT ===
Goal: Minimize total_cost = (lataccel_cost * 50) + jerk_cost for lateral car control.
Simulator: vendor/commaai/tinyphysics.py (ONNX model, CPU-only currently)
GPU: RTX 5070 Ti (16GB VRAM) — should be used for training
PID baseline: ~81 on 100 segments
SOTA: tfpgh v2 = 17.789 (MPC with inverse CDF sampling)
Project root: ${PROJECT_ROOT}

=== RULES ===
- DO NOT edit code, write scripts, or modify files in workspace/ or vendor/. That is the orchestrator's job.
  You are READ-ONLY for workspace and vendor files. The only file you write to is ${COMMANDS_FILE}.
- NEVER start long-running python scripts. Relay to orchestrator via commands file.
- Quick system commands are OK: nvidia-smi, ps aux, kill PID, tail logs, pip list, etc.
- When relaying to orchestrator, be specific: "Implement X using Y approach, evaluate on 100 segments"
- If the user asks you to do something the orchestrator should do, relay it.
- If it's a quick check you can do yourself (status, scores, logs), just do it.
- Keep responses under 1500 chars.`;

const MEDIATOR_SESSION_OPTIONS: any = {
  model: "claude-opus-4-6",
  allowedTools: ["Read", "Bash", "Glob", "Grep"],
  permissionMode: "bypassPermissions",
  allowDangerouslySkipPermissions: true,
};

let mediatorSessionId: string | null = null;
let mediatorBusy = false;
let messageQueue: Array<{ content: string; message: Message }> = [];

function extractText(msg: SDKMessage): string {
  if (msg.type !== "assistant") return "";
  const m = msg.message as any;
  if (!m?.content) return "";
  if (Array.isArray(m.content)) {
    return m.content
      .filter((b: any) => b.type === "text")
      .map((b: any) => b.text)
      .join("\n");
  }
  return String(m.content);
}

// ===== Live Discord Output =====

/** Manages a live-updating Discord message that mirrors Claude Code output */
class LiveOutput {
  private channel: TextChannel;
  private messages: Message[] = [];
  private currentContent = "";
  private pendingFlush: NodeJS.Timeout | null = null;
  private toolLines: string[] = [];
  private responseText = "";
  private mention: string;

  constructor(channel: TextChannel) {
    this.channel = channel;
    this.mention = USER_ID ? `<@${USER_ID}> ` : "";
  }

  /** Add a tool activity line (e.g., "Reading workspace/results.json...") */
  addToolActivity(line: string) {
    this.toolLines.push(line);
    this.scheduleFlush();
  }

  /** Set the assistant's text response */
  setResponse(text: string) {
    this.responseText = text;
    this.scheduleFlush();
  }

  /** Append to assistant's text response */
  appendResponse(text: string) {
    this.responseText += text;
    this.scheduleFlush();
  }

  private scheduleFlush() {
    if (this.pendingFlush) return;
    this.pendingFlush = setTimeout(() => {
      this.pendingFlush = null;
      this.flush();
    }, 800); // Batch updates to avoid Discord rate limits
  }

  private async flush() {
    // Build the current display
    let display = this.mention + "**[mediator]** ";

    // Show tool activity as dim/code block
    if (this.toolLines.length > 0) {
      const recent = this.toolLines.slice(-5);
      display += "\n```\n" + recent.join("\n") + "\n```\n";
    }

    // Show response text
    if (this.responseText) {
      display += this.responseText;
    }

    if (!this.toolLines.length && !this.responseText) {
      display += "_thinking..._";
    }

    // Truncate to Discord limit
    if (display.length > 1990) {
      display = display.slice(0, 1987) + "...";
    }

    try {
      if (this.messages.length === 0) {
        // Send first message
        const msg = await this.channel.send(display);
        this.messages.push(msg);
        this.currentContent = display;
      } else if (display !== this.currentContent) {
        // Edit existing message
        await this.messages[this.messages.length - 1].edit(display);
        this.currentContent = display;
      }
    } catch (err) {
      console.error("[live-output] flush error:", err);
    }
  }

  /** Final flush — ensure everything is sent */
  async finalize() {
    if (this.pendingFlush) {
      clearTimeout(this.pendingFlush);
      this.pendingFlush = null;
    }
    await this.flush();

    // If response is very long, send continuation messages
    if (this.responseText.length > 1800) {
      const fullText = this.mention + this.responseText;
      const chunks = fullText.match(/[\s\S]{1,1990}/g) || [];
      // First chunk already sent via edit, send remaining
      for (let i = 1; i < chunks.length; i++) {
        try {
          await this.channel.send(chunks[i]);
        } catch {}
      }
    }
  }
}

/** Run a single mediator turn with live output to Discord */
async function mediatorTurn(userMessage: string, liveOutput: LiveOutput): Promise<void> {
  let session: SDKSession;

  if (mediatorSessionId) {
    console.log(`[mediator] Resuming session ${mediatorSessionId.slice(0, 8)}...`);
    session = unstable_v2_resumeSession(mediatorSessionId, MEDIATOR_SESSION_OPTIONS);
    await session.send(userMessage);
  } else {
    console.log("[mediator] Creating new session...");
    session = unstable_v2_createSession(MEDIATOR_SESSION_OPTIONS);
    await session.send(`${MEDIATOR_SYSTEM}\n\n---\n\nUser message: ${userMessage}`);
  }

  const stream = session.stream();

  for await (const msg of stream) {
    if (msg.type === "result") {
      if (!mediatorSessionId) {
        try { mediatorSessionId = session.sessionId; } catch {}
      }
      break;
    }

    switch (msg.type) {
      case "tool_use_summary": {
        // Show tool activity like Claude Code does
        const summary = (msg as any).summary as string;
        if (summary) {
          liveOutput.addToolActivity(summary);
          console.log(`[mediator:tool] ${summary}`);
        }
        break;
      }

      case "tool_progress": {
        const tp = msg as any;
        const toolName = tp.tool_name || "tool";
        const elapsed = tp.elapsed_time_seconds || 0;
        if (elapsed > 2) {
          liveOutput.addToolActivity(`${toolName}... (${elapsed}s)`);
        }
        break;
      }

      case "assistant": {
        const text = extractText(msg);
        if (text) {
          liveOutput.setResponse(text);
          console.log(`[mediator] ${text.slice(0, 200)}`);
        }
        break;
      }
    }
  }

  if (!mediatorSessionId) {
    try { mediatorSessionId = session.sessionId; } catch {}
  }
}

async function sendToMediator(userContent: string, discordMessage: Message): Promise<void> {
  const channel = discordMessage.channel as TextChannel;

  if (mediatorBusy) {
    messageQueue.push({ content: userContent, message: discordMessage });
    await discordMessage.reply("Processing previous message, yours is queued.");
    return;
  }

  mediatorBusy = true;
  const liveOutput = new LiveOutput(channel);

  try {
    await mediatorTurn(userContent, liveOutput);
    await liveOutput.finalize();
  } catch (err) {
    console.error("[mediator] Error:", err);
    const errMsg = String(err);

    if (
      errMsg.includes("closed") ||
      errMsg.includes("disposed") ||
      errMsg.includes("EPIPE") ||
      errMsg.includes("session")
    ) {
      console.log("[mediator] Session error, will create new session on next message");
      mediatorSessionId = null;
    }

    await liveOutput.finalize();
    await channel.send(
      `<@${USER_ID}> Mediator error: ${errMsg.slice(0, 200)}. Will retry on next message.`
    );
  } finally {
    mediatorBusy = false;

    if (messageQueue.length > 0) {
      const next = messageQueue.shift()!;
      sendToMediator(next.content, next.message);
    }
  }
}

// ===== Discord Client =====

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

const channels: Record<string, TextChannel> = {};

client.once(Events.ClientReady, (c) => {
  console.log(`Discord bot online as ${c.user.tag}`);

  const vibes = client.channels.cache.get(VIBES_CHANNEL_ID!) as TextChannel;
  if (vibes) channels.vibes = vibes;

  if (RLCLAW_CHANNEL_ID) {
    const rlclaw = client.channels.cache.get(RLCLAW_CHANNEL_ID) as TextChannel;
    if (rlclaw) channels.rlclaw = rlclaw;
  }

  watchOutbox();
});

// Listen for @mentions from authorized user
client.on(Events.MessageCreate, async (message) => {
  if (message.author.bot) return;
  if (message.channelId !== VIBES_CHANNEL_ID) return;
  if (!message.mentions.has(client.user!.id)) return;

  if (message.author.id !== USER_ID) {
    await message.reply("not authorized");
    return;
  }

  const content = message.content.replace(/<@!?\d+>/g, "").trim();
  if (!content) {
    await message.reply("send a command after the mention");
    return;
  }

  console.log(`[${new Date().toISOString()}] User: ${content.slice(0, 100)}`);

  // Handle instant bot commands (no mediator needed)
  if (content === "!help") {
    await message.reply(
      "**Commands:**\n" +
        "`!status` — quick status (no mediator)\n" +
        "`!resume` — resume orchestrator autonomous work\n" +
        "`!reset` — reset mediator session (fresh context)\n" +
        "`!help` — this message\n" +
        "Anything else — goes to the mediator (Opus), which can answer or steer the orchestrator"
    );
    return;
  }

  if (content === "!resume") {
    fs.writeFileSync(COMMANDS_FILE, "!resume");
    await message.reply("Resuming orchestrator autonomous work.");
    return;
  }

  if (content === "!reset") {
    mediatorSessionId = null;
    await message.reply("Mediator session reset. Next message starts fresh.");
    return;
  }

  if (content === "!status") {
    const status = getQuickStatus();
    await message.reply(status);
    return;
  }

  // Everything else goes to the mediator
  await sendToMediator(content, message);
});

/** Build a quick status summary from local files */
function getQuickStatus(): string {
  const lines: string[] = [];

  try {
    const entries = fs.readFileSync(SESSION_LOG, "utf-8").trim().split("\n");
    for (let i = entries.length - 1; i >= 0; i--) {
      try {
        const e = JSON.parse(entries[i]);
        if (e.role === "orchestrator") {
          lines.push(`**Last action:** ${e.content.slice(0, 200)}`);
          break;
        }
      } catch {}
    }
  } catch {}

  try {
    const results = JSON.parse(fs.readFileSync(RESULTS_PATH, "utf-8"));
    const exps = results.experiments || [];
    if (exps.length > 0) {
      const best = exps.reduce((a: any, b: any) =>
        a.total_cost < b.total_cost ? a : b
      );
      lines.push(`**Best score:** ${best.total_cost.toFixed(2)} (${best.controller})`);
    }
  } catch {}

  try {
    const telem = JSON.parse(fs.readFileSync(TELEMETRY_PATH, "utf-8"));
    lines.push(
      `**Turns:** ${telem.turns || 0} | **Cost:** $${(telem.costEstimate || 0).toFixed(2)}`
    );
  } catch {}

  return lines.join("\n") || "No status available";
}

// Watch outbox dir for proactive messages from the orchestrator/notify system
function watchOutbox() {
  setInterval(async () => {
    try {
      if (!fs.existsSync(OUTBOX_DIR)) return;
      const files = fs.readdirSync(OUTBOX_DIR).sort();
      for (const file of files) {
        const filePath = path.join(OUTBOX_DIR, file);
        const content = fs.readFileSync(filePath, "utf-8").trim();
        fs.unlinkSync(filePath);

        if (!content) continue;

        const channelName = file.includes("_rlclaw") ? "rlclaw" : "vibes";
        const channel = channels[channelName];
        if (!channel) continue;

        const chunks = content.match(/[\s\S]{1,1990}/g) || [content];
        for (const chunk of chunks) {
          await channel.send(chunk);
        }
      }
    } catch {}
  }, 2_000);
}

client.login(TOKEN);
