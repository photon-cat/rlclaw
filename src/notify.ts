import * as fs from "fs";
import * as path from "path";
import { recordDiscord } from "./telemetry";

const OUTBOX_DIR = path.join(__dirname, "discord_outbox");

export async function notify(
  content: string,
  level: "info" | "success" | "error" | "input" = "info",
  channel: "vibes" | "rlclaw" = "vibes"
): Promise<void> {
  const mention = process.env.DISCORD_USER_ID ? `<@${process.env.DISCORD_USER_ID}>` : "";
  const ping = (level === "input" || level === "error") && mention ? ` ${mention}` : "";
  const msg = `**[orchestrator]**${ping} ${content}`;
  const truncated = msg.length > 1990 ? msg.slice(0, 1990) + "..." : msg;

  try { recordDiscord(channel, truncated); } catch {}

  // Write to outbox for the bot to pick up and send
  try {
    fs.mkdirSync(OUTBOX_DIR, { recursive: true });
    const filename = `${Date.now()}_${channel}.txt`;
    fs.writeFileSync(path.join(OUTBOX_DIR, filename), truncated);
  } catch {
    console.error("[notify] Failed to write to outbox");
  }
}
