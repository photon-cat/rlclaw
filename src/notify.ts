function getConfig() {
  return {
    webhook: process.env.DISCORD_WEBHOOK,
    mention: process.env.DISCORD_USER_ID ? `<@${process.env.DISCORD_USER_ID}>` : "",
  };
}

export async function notify(
  content: string,
  level: "info" | "success" | "error" | "input" = "info"
): Promise<void> {
  const { webhook, mention } = getConfig();
  if (!webhook) {
    console.error("[notify] DISCORD_WEBHOOK not set in .env");
    return;
  }

  const prefix =
    level === "success"
      ? ":white_check_mark:"
      : level === "error"
        ? ":x:"
        : level === "input"
          ? ":question:"
          : ":brain:";

  const ping = (level === "input" || level === "error") && mention ? ` ${mention}` : "";
  const msg = `${prefix} **rlclaw**${ping} — ${content}`;

  const truncated = msg.length > 1990 ? msg.slice(0, 1990) + "..." : msg;

  try {
    await fetch(webhook, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: truncated }),
    });
  } catch {
    console.error("[notify] Discord webhook failed");
  }
}
