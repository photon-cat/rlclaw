#!/usr/bin/env npx tsx
/**
 * colab — CLI for managing Colab GPU runtimes and executing code.
 *
 * Usage:
 *   colab auth                          Sign in with Google
 *   colab info                          Account info + available GPUs
 *   colab ls                            List active runtimes
 *   colab create [type] [accelerator]   Create runtime (GPU T4, GPU A100, TPU, etc)
 *   colab rm [endpoint]                 Disconnect a runtime
 *   colab exec <code>                   Execute Python code
 *   colab exec -f <file.py>             Execute a Python file
 *   colab run <file.py>                 Upload and run a Python file
 *   colab kernels                       List kernels on all runtimes
 *   colab shell                         Interactive Python shell
 */

import { ColabClient, ColabRuntime, ColabConnection } from "./client";
import * as fs from "fs";
import * as readline from "readline";

const client = new ColabClient();

// --- Helpers ---

function die(msg: string): never {
  console.error(`Error: ${msg}`);
  process.exit(1);
}

function runtimeLabel(r: ColabRuntime): string {
  const gpu = r.accelerator !== "NONE" ? r.accelerator : "CPU";
  return `${r.endpoint} [${gpu}]`;
}

async function getGpuRuntime(): Promise<ColabRuntime> {
  const runtimes = await client.listRuntimes();
  const gpu = runtimes.find((r) => r.accelerator !== "NONE");
  if (gpu) return gpu;
  if (runtimes.length) return runtimes[0];
  die("No runtimes. Create one with: colab create GPU T4");
}

async function getConnection(runtime?: ColabRuntime): Promise<ColabConnection> {
  const rt = runtime || (await getGpuRuntime());
  return client.connect(rt);
}

// --- Commands ---

async function cmdAuth() {
  client.quiet = false;
  await client.auth();
  console.log("Authenticated.");
}

async function cmdInfo() {
  const info = await client.getUserInfo();
  console.log(`Subscription: ${info.subscriptionTier}`);
  console.log(`Compute units: ${info.paidComputeUnitsBalance.toFixed(1)}`);
  console.log(`\nAvailable accelerators:`);
  for (const group of info.eligibleAccelerators) {
    const variant = group.variant.replace("VARIANT_", "");
    console.log(`  ${variant}: ${group.models.join(", ")}`);
  }
}

async function cmdLs() {
  const runtimes = await client.listRuntimes();
  if (!runtimes.length) {
    console.log("No active runtimes.");
    return;
  }
  console.log(`${runtimes.length} runtime(s):\n`);
  for (const r of runtimes) {
    const gpu = r.accelerator !== "NONE" ? r.accelerator : "CPU";
    const shape = r.machineShape !== "SHAPE_DEFAULT" ? ` (${r.machineShape})` : "";
    console.log(`  ${r.endpoint}`);
    console.log(`    type: ${gpu}${shape}`);
    if (r.baseUrl) console.log(`    url:  ${r.baseUrl}`);
    console.log();
  }
}

async function cmdCreate(args: string[]) {
  const variant = (args[0]?.toUpperCase() || "GPU") as "DEFAULT" | "GPU" | "TPU";
  const accelerator = args[1]?.toUpperCase();
  const runtime = await client.createRuntime(variant, accelerator);
  const conn = await client.connect(runtime);

  console.log(`\nRuntime ready:`);
  console.log(`  endpoint: ${runtime.endpoint}`);
  console.log(`  type:     ${runtime.accelerator}`);
  console.log(`  url:      ${conn.baseUrl}`);

  // Start a kernel so it's ready for exec
  console.log(`\nStarting kernel...`);
  const kid = await client.startKernel(conn);
  console.log(`  kernel: ${kid}`);
  console.log(`\nReady. Run: colab exec 'print("hello")'`);
}

async function cmdExec(args: string[]) {
  let code: string;

  if (args[0] === "-f" && args[1]) {
    code = fs.readFileSync(args[1], "utf-8");
  } else if (args.length) {
    code = args.join(" ");
  } else {
    // Read from stdin
    code = fs.readFileSync(0, "utf-8");
  }

  const conn = await getConnection();
  const result = await client.execute(conn, code);

  for (const o of result.outputs) process.stdout.write(o);
  if (result.error) {
    process.exitCode = 1;
  }
}

async function cmdRun(args: string[]) {
  const file = args[0];
  if (!file) die("Usage: colab run <file.py>");
  if (!fs.existsSync(file)) die(`File not found: ${file}`);

  const code = fs.readFileSync(file, "utf-8");
  const conn = await getConnection();

  console.error(`Running ${file} on ${conn.runtime.endpoint}...`);
  const result = await client.execute(conn, code);

  for (const o of result.outputs) process.stdout.write(o);
  if (result.error) {
    process.exitCode = 1;
  }
}

async function cmdKernels() {
  const runtimes = await client.listRuntimes();
  if (!runtimes.length) {
    console.log("No runtimes.");
    return;
  }
  for (const r of runtimes) {
    console.log(`\n${runtimeLabel(r)}:`);
    try {
      const conn = await client.connect(r);
      const kernels = await client.listKernels(conn);
      if (!kernels.length) {
        console.log("  (no kernels)");
        continue;
      }
      for (const k of kernels) {
        console.log(`  ${k.id}  ${k.execution_state}  conns=${k.connections}`);
      }
    } catch (e: any) {
      console.log(`  error: ${e.message}`);
    }
  }
}

async function cmdShell() {
  const conn = await getConnection();
  console.log(`Connected to ${conn.runtime.endpoint}`);
  console.log(`Type Python code. Ctrl+D to exit.\n`);

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: ">>> ",
  });

  rl.prompt();
  let buffer = "";

  rl.on("line", async (line) => {
    buffer += line + "\n";

    // Simple heuristic: execute on blank line after content, or single-line expressions
    if (line.trim() === "" && buffer.trim() !== "") {
      const code = buffer;
      buffer = "";
      try {
        const result = await client.execute(conn, code);
        for (const o of result.outputs) process.stdout.write(o);
      } catch (e: any) {
        console.error(`Error: ${e.message}`);
      }
      rl.prompt();
    } else if (!line.startsWith(" ") && !line.endsWith(":") && !line.endsWith("\\")) {
      // Single complete line
      const code = buffer;
      buffer = "";
      try {
        const result = await client.execute(conn, code);
        for (const o of result.outputs) process.stdout.write(o);
      } catch (e: any) {
        console.error(`Error: ${e.message}`);
      }
      rl.prompt();
    } else {
      process.stdout.write("... ");
    }
  });

  rl.on("close", () => process.exit(0));
}

// --- Main ---

async function main() {
  const args = process.argv.slice(2);
  const cmd = args[0] || "help";
  const rest = args.slice(1);

  // Auth first for all commands except help/auth
  if (cmd !== "help" && cmd !== "--help" && cmd !== "-h") {
    client.quiet = true;
    await client.auth();
  }

  switch (cmd) {
    case "auth":    return cmdAuth();
    case "info":    return cmdInfo();
    case "ls":
    case "list":    return cmdLs();
    case "create":
    case "new":     return cmdCreate(rest);
    case "exec":    return cmdExec(rest);
    case "run":     return cmdRun(rest);
    case "kernels": return cmdKernels();
    case "shell":
    case "repl":    return cmdShell();
    default:
      console.log(`colab — Colab GPU runtime CLI

Usage:
  colab auth                          Sign in with Google
  colab info                          Account info, compute units, available GPUs
  colab ls                            List active runtimes
  colab create [GPU|TPU] [model]      Create a runtime (e.g. create GPU T4)
  colab exec '<python code>'          Execute Python code on GPU
  colab exec -f script.py             Execute a Python file on GPU
  colab run script.py                 Run a Python file (output to stdout)
  colab kernels                       List kernels on all runtimes
  colab shell                         Interactive Python REPL on GPU

Examples:
  colab create GPU T4
  colab exec 'import torch; print(torch.cuda.get_device_name(0))'
  colab exec -f train.py
  echo 'print("hi")' | colab exec`);
  }
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
