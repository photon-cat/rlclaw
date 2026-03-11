/**
 * Standalone Colab client — no VS Code dependency.
 *
 * Authenticates via Google OAuth (browser flow), then talks directly
 * to the Colab GAPI and Jupyter kernel REST API/WebSocket.
 */

import * as http from "http";
import * as https from "https";
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { URL, URLSearchParams } from "url";
import WebSocket from "ws";

// Load .env from package root
const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, "utf-8").split("\n")) {
    const match = line.match(/^([^#=]+)=(.*)$/);
    if (match && !process.env[match[1].trim()]) {
      process.env[match[1].trim()] = match[2].trim();
    }
  }
}

const CLIENT_ID = process.env.COLAB_CLIENT_ID || "";
const CLIENT_SECRET = process.env.COLAB_CLIENT_SECRET || "";
const SCOPES = [
  "profile",
  "email",
  "https://www.googleapis.com/auth/colaboratory",
];
const COLAB_GAPI = "https://colab.pa.googleapis.com";
const COLAB_DOMAIN = "https://colab.research.google.com";
const TOKEN_FILE = path.join(
  __dirname, "..",
  ".google_token.json"
);

export interface ColabRuntime {
  endpoint: string;
  variant: string;
  accelerator: string;
  machineShape: string;
  baseUrl?: string;
  proxyToken?: string;
}

export interface ColabConnection {
  runtime: ColabRuntime;
  baseUrl: string;
  token: string;
  expiresIn: number;
}

export interface ExecResult {
  outputs: string[];
  error?: string;
}

export interface UserInfo {
  subscriptionTier: string;
  paidComputeUnitsBalance: number;
  eligibleAccelerators: { variant: string; models: string[] }[];
}

interface TokenData {
  access_token: string;
  refresh_token?: string;
  expires_at: number;
}

// --- HTTP ---

const COLAB_RESPONSE_PREFIX = ")]}'\n";

function stripPrefix(text: string): string {
  return text.startsWith(COLAB_RESPONSE_PREFIX)
    ? text.slice(COLAB_RESPONSE_PREFIX.length)
    : text;
}

function httpsJson(
  url: string | URL,
  options: https.RequestOptions & { body?: string } = {}
): Promise<any> {
  return new Promise((resolve, reject) => {
    const req = https.request(url, options, (res) => {
      let data = "";
      res.on("data", (chunk: string) => (data += chunk));
      res.on("end", () => {
        if (res.statusCode && res.statusCode >= 400) {
          reject(new Error(`HTTP ${res.statusCode}: ${data.slice(0, 500)}`));
        } else {
          try {
            resolve(JSON.parse(stripPrefix(data)));
          } catch {
            resolve(data);
          }
        }
      });
    });
    req.on("error", reject);
    if (options.body) req.write(options.body);
    req.end();
  });
}

// --- OAuth ---

function loadToken(): TokenData | null {
  try {
    return JSON.parse(fs.readFileSync(TOKEN_FILE, "utf-8"));
  } catch {
    return null;
  }
}

function saveToken(data: TokenData) {
  fs.mkdirSync(path.dirname(TOKEN_FILE), { recursive: true });
  fs.writeFileSync(TOKEN_FILE, JSON.stringify(data, null, 2), { mode: 0o600 });
}

async function exchangeCode(code: string, redirectUri: string): Promise<TokenData> {
  const result = await httpsJson("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      code, client_id: CLIENT_ID, client_secret: CLIENT_SECRET,
      redirect_uri: redirectUri, grant_type: "authorization_code",
    }).toString(),
  });
  return {
    access_token: result.access_token,
    refresh_token: result.refresh_token,
    expires_at: Date.now() + result.expires_in * 1000,
  };
}

async function refreshToken(rt: string): Promise<TokenData> {
  const result = await httpsJson("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      refresh_token: rt, client_id: CLIENT_ID, client_secret: CLIENT_SECRET,
      grant_type: "refresh_token",
    }).toString(),
  });
  return {
    access_token: result.access_token,
    refresh_token: rt,
    expires_at: Date.now() + result.expires_in * 1000,
  };
}

async function browserAuthFlow(): Promise<TokenData> {
  return new Promise((resolve, reject) => {
    const srv = http.createServer(async (req, res) => {
      try {
        const code = new URL(req.url!, "http://localhost").searchParams.get("code");
        if (!code) { res.writeHead(400); res.end("Missing code"); return; }
        res.writeHead(200, { "Content-Type": "text/html" });
        res.end("<html><body><h2>Authenticated! You can close this tab.</h2></body></html>");
        const addr = srv.address() as { port: number };
        const token = await exchangeCode(code, `http://localhost:${addr.port}`);
        srv.close();
        resolve(token);
      } catch (err) { srv.close(); reject(err); }
    });

    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address() as { port: number };
      const params = new URLSearchParams({
        client_id: CLIENT_ID, redirect_uri: `http://localhost:${addr.port}`,
        response_type: "code", scope: SCOPES.join(" "),
        access_type: "offline", prompt: "consent",
      });
      const url = `https://accounts.google.com/o/oauth2/v2/auth?${params}`;
      const { exec } = require("child_process");
      const open = process.platform === "darwin" ? "open" : process.platform === "win32" ? "start" : "xdg-open";
      exec(`${open} '${url}'`);
    });

    setTimeout(() => { srv.close(); reject(new Error("Auth timeout")); }, 120000);
  });
}

// --- Client ---

export class ColabClient {
  private token: TokenData | null = null;
  quiet = false;

  private log(...args: any[]) {
    if (!this.quiet) console.error(...args);
  }

  async auth(): Promise<void> {
    if (!CLIENT_ID || !CLIENT_SECRET) {
      throw new Error(
        "Missing COLAB_CLIENT_ID and/or COLAB_CLIENT_SECRET env vars.\n" +
        "Extract them from the Colab VS Code extension:\n" +
        "  grep -o 'apps.googleusercontent.com' ~/.vscode/extensions/google.colab-*/out/extension.js\n" +
        "See README for details."
      );
    }
    this.token = loadToken();
    if (this.token?.refresh_token && this.token.expires_at < Date.now()) {
      try {
        this.token = await refreshToken(this.token.refresh_token);
        saveToken(this.token);
        return;
      } catch { /* fall through to browser auth */ }
    }
    if (this.token && this.token.expires_at > Date.now()) return;
    this.log("Opening browser for Google auth...");
    this.token = await browserAuthFlow();
    saveToken(this.token);
    this.log("Authenticated.");
  }

  private async accessToken(): Promise<string> {
    if (!this.token) throw new Error("Not authenticated — call auth() first");
    if (this.token.expires_at < Date.now() && this.token.refresh_token) {
      this.token = await refreshToken(this.token.refresh_token);
      saveToken(this.token);
    }
    return this.token.access_token;
  }

  private async gapiGet(path: string): Promise<any> {
    return httpsJson(new URL(path, COLAB_GAPI), {
      method: "GET",
      headers: {
        Authorization: `Bearer ${await this.accessToken()}`,
        "X-Colab-Client-Agent": "rlclaw",
      },
    });
  }

  async getUserInfo(): Promise<UserInfo> {
    return this.gapiGet("/v1/user-info");
  }

  async listRuntimes(): Promise<ColabRuntime[]> {
    const data = await this.gapiGet("/v1/assignments");
    return (data.assignments || []).map((a: any) => ({
      endpoint: a.endpoint,
      variant: a.variant || "UNKNOWN",
      accelerator: a.accelerator || "NONE",
      machineShape: a.machineShape || "DEFAULT",
      baseUrl: a.runtimeProxyInfo?.url,
      proxyToken: a.runtimeProxyInfo?.token,
    }));
  }

  async connect(runtime: ColabRuntime): Promise<ColabConnection> {
    // If we already have proxy info from listRuntimes, use it
    if (runtime.baseUrl && runtime.proxyToken) {
      return {
        runtime,
        baseUrl: runtime.baseUrl,
        token: runtime.proxyToken,
        expiresIn: 3600,
      };
    }
    const data = await httpsJson(
      (() => {
        const u = new URL("/v1/runtime-proxy-token", COLAB_GAPI);
        u.searchParams.set("endpoint", runtime.endpoint);
        u.searchParams.set("port", "8080");
        return u;
      })(),
      {
        method: "GET",
        headers: {
          Authorization: `Bearer ${await this.accessToken()}`,
          "X-Colab-Client-Agent": "rlclaw",
        },
      }
    );
    return {
      runtime,
      baseUrl: data.url,
      token: data.token,
      expiresIn: parseInt((data.tokenTtl || "3600s").replace("s", ""), 10),
    };
  }

  async createRuntime(
    variant: "DEFAULT" | "GPU" | "TPU" = "GPU",
    accelerator?: string
  ): Promise<ColabRuntime> {
    const token = await this.accessToken();
    const nbh = crypto.randomUUID().replace(/-/g, "_") + ".".repeat(8);
    const assignUrl = new URL("/tun/m/assign", COLAB_DOMAIN);
    assignUrl.searchParams.set("authuser", "0");
    assignUrl.searchParams.set("nbh", nbh);
    if (variant !== "DEFAULT") assignUrl.searchParams.set("variant", variant);
    if (accelerator) assignUrl.searchParams.set("accelerator", accelerator);

    const headers = {
      Authorization: `Bearer ${token}`,
      Accept: "application/json",
      "X-Colab-Client-Agent": "rlclaw",
    };

    this.log(`Creating ${variant}${accelerator ? ` ${accelerator}` : ""} runtime...`);
    const check = await httpsJson(assignUrl, { method: "GET", headers });

    if (check.endpoint) {
      this.log(`Already assigned: ${check.endpoint}`);
      return {
        endpoint: check.endpoint,
        variant, accelerator: check.accelerator || accelerator || "NONE",
        machineShape: check.machineShape || "DEFAULT",
        baseUrl: check.runtimeProxyInfo?.url,
        proxyToken: check.runtimeProxyInfo?.token,
      };
    }

    if (!check.token) throw new Error(`Unexpected: ${JSON.stringify(check).slice(0, 200)}`);

    const created = await httpsJson(assignUrl, {
      method: "POST",
      headers: { ...headers, "X-Goog-Colab-Token": check.token },
    });

    if (!created.endpoint) throw new Error(`Create failed: ${JSON.stringify(created).slice(0, 300)}`);
    this.log(`Created: ${created.endpoint}`);

    return {
      endpoint: created.endpoint,
      variant, accelerator: created.accelerator || accelerator || "NONE",
      machineShape: created.machineShape || "DEFAULT",
      baseUrl: created.runtimeProxyInfo?.url,
      proxyToken: created.runtimeProxyInfo?.token,
    };
  }

  async startKernel(conn: ColabConnection): Promise<string> {
    const result = await httpsJson(new URL("/api/kernels", conn.baseUrl), {
      method: "POST",
      headers: {
        "X-Colab-Runtime-Proxy-Token": conn.token,
        "X-Colab-Client-Agent": "rlclaw",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name: "python3" }),
    });
    return result.id;
  }

  async listKernels(conn: ColabConnection): Promise<any[]> {
    return httpsJson(new URL("/api/kernels", conn.baseUrl), {
      method: "GET",
      headers: {
        "X-Colab-Runtime-Proxy-Token": conn.token,
        "X-Colab-Client-Agent": "rlclaw",
      },
    });
  }

  async execute(conn: ColabConnection, code: string, timeoutMs = 300000): Promise<ExecResult> {
    const headers: Record<string, string> = {
      "X-Colab-Runtime-Proxy-Token": conn.token,
      "X-Colab-Client-Agent": "rlclaw",
    };

    let kernels = await httpsJson(new URL("/api/kernels", conn.baseUrl), {
      method: "GET", headers,
    });

    if (!kernels.length) {
      this.log("Starting kernel...");
      await this.startKernel(conn);
      await new Promise((r) => setTimeout(r, 5000));
      kernels = await httpsJson(new URL("/api/kernels", conn.baseUrl), {
        method: "GET", headers,
      });
    }

    const idle = kernels.find((k: any) => k.execution_state === "idle");
    const kernelId = (idle || kernels[0]).id;
    const sessionId = `rlclaw-${Date.now()}`;
    const wsUrl = conn.baseUrl.replace("https:", "wss:") +
      `/api/kernels/${kernelId}/channels?session_id=${sessionId}`;

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl, { headers });
      const msgId = `exec-${Date.now()}`;
      const outputs: string[] = [];
      let error: string | undefined;
      const timer = setTimeout(() => { ws.close(); reject(new Error("Execution timeout")); }, timeoutMs);

      ws.on("open", () => {
        ws.send(JSON.stringify({
          header: {
            msg_id: msgId, msg_type: "execute_request", username: "rlclaw",
            session: sessionId, date: new Date().toISOString(), version: "5.3",
          },
          parent_header: {}, metadata: {},
          content: { code, silent: false, store_history: false, user_expressions: {}, allow_stdin: false, stop_on_error: true },
          channel: "shell",
        }));
      });

      ws.on("message", (data: WebSocket.Data) => {
        try {
          const msg = JSON.parse(data.toString());
          if (msg.parent_header?.msg_id !== msgId) return;
          switch (msg.msg_type) {
            case "stream": outputs.push(msg.content.text); break;
            case "execute_result":
            case "display_data":
              if (msg.content?.data?.["text/plain"]) outputs.push(msg.content.data["text/plain"]);
              break;
            case "error":
              error = `${msg.content.ename}: ${msg.content.evalue}`;
              outputs.push(msg.content.traceback?.join("\n") || error);
              break;
            case "execute_reply":
              clearTimeout(timer); ws.close(); resolve({ outputs, error });
              break;
          }
        } catch { /* binary */ }
      });

      ws.on("error", (err) => { clearTimeout(timer); reject(err); });
    });
  }

  /** High-level: connect to a runtime by index and execute code */
  async run(code: string, runtimeIndex = 0): Promise<ExecResult> {
    const runtimes = await this.listRuntimes();
    if (!runtimes.length) throw new Error("No runtimes assigned");
    const conn = await this.connect(runtimes[runtimeIndex] || runtimes[0]);
    return this.execute(conn, code);
  }
}
