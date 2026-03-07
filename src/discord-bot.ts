import { Client, GatewayIntentBits, Events } from "discord.js";
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
const USER_ID = process.env.DISCORD_USER_ID;
const COMMANDS_FILE = path.join(__dirname, "..", "commands.txt");

if (!TOKEN || !VIBES_CHANNEL_ID) {
  console.error("Missing DISCORD_BOT_TOKEN or DISCORD_VIBES_CHANNEL_ID in .env");
  process.exit(1);
}

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

client.once(Events.ClientReady, (c) => {
  console.log(`Discord bot online as ${c.user.tag}`);
  console.log(`Listening in #vibes (${VIBES_CHANNEL_ID}) for messages from user ${USER_ID}`);
});

client.on(Events.MessageCreate, async (message) => {
  // Ignore bots and messages outside vibes channel
  if (message.author.bot) return;
  if (message.channelId !== VIBES_CHANNEL_ID) return;

  // Only accept messages from the authorized user
  if (message.author.id !== USER_ID) return;

  const content = message.content.trim();
  if (!content) return;

  // Write to commands.txt for the orchestrator to pick up
  fs.writeFileSync(COMMANDS_FILE, content);
  console.log(`[${new Date().toISOString()}] Command from Discord: ${content.slice(0, 100)}`);

  // React to confirm receipt
  try {
    await message.react("👍");
  } catch {}
});

client.login(TOKEN);
