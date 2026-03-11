# colab-cli

CLI tool for managing Google Colab GPU runtimes and executing Python code from the terminal. No browser or VS Code needed after initial auth.

## Setup

### 1. Install the Colab VS Code extension

Install [Google Colab](https://marketplace.visualstudio.com/items?itemName=google.colab) in VS Code — you need it installed, not running.

### 2. Extract OAuth credentials

```bash
# Client ID
grep -oE '[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com' \
  ~/.vscode/extensions/google.colab-*/out/extension.js | head -1

# Client Secret
grep -oE 'GOCSPX-[A-Za-z0-9_-]+' \
  ~/.vscode/extensions/google.colab-*/out/extension.js | head -1
```

### 3. Create `.env` file

```bash
# In the colab-cli directory
cat > .env << 'EOF'
COLAB_CLIENT_ID=<client-id-from-step-2>
COLAB_CLIENT_SECRET=<secret-from-step-2>
EOF
```

Or export them in your shell (`~/.zshrc` / `~/.bashrc`):
```bash
export COLAB_CLIENT_ID="<client-id-from-step-2>"
export COLAB_CLIENT_SECRET="<secret-from-step-2>"
```

### 4. Install

```bash
npm install
```

## Quick Start

```bash
# 1. Authenticate (opens browser, one-time)
npx colab auth

# 2. Check your account
npx colab info
# → Subscription: PRO_PLUS
# → Compute units: 574.0
# → Available accelerators: GPU: T4, A100, L4, H100 ...

# 3. Spin up a GPU runtime
npx colab create GPU T4

# 4. Run code on it
npx colab exec 'import torch; print(torch.cuda.get_device_name(0))'
# → Tesla T4
```

## Usage

### Managing runtimes

```bash
# Create runtimes — spins up in ~30-60s
npx colab create              # Default (CPU)
npx colab create GPU T4       # T4 GPU (free tier)
npx colab create GPU A100     # A100 GPU (Pro/Pro+)
npx colab create GPU H100     # H100 GPU (Pro+ only)
npx colab create TPU          # TPU

# List active runtimes
npx colab ls
# → abc123def [T4]
# →   type: NVIDIA_TESLA_T4
# →   url:  https://...

# List kernels on all runtimes
npx colab kernels
```

### Executing code

```bash
# Inline Python
npx colab exec 'print("hello from GPU")'

# Multi-line
npx colab exec 'import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")'

# From a file
npx colab exec -f train.py

# Pipe from stdin
echo 'print("hi")' | npx colab exec
cat train.py | npx colab exec

# Run a file (output to stdout, errors to stderr)
npx colab run train.py
```

### Interactive REPL

```bash
npx colab shell
# >>> import torch
# >>> torch.cuda.get_device_name(0)
# 'Tesla T4'
# >>> ^D (Ctrl+D to exit)
```

### Account info

```bash
npx colab info
# → Subscription: PRO_PLUS
# → Compute units: 574.0
# → Available accelerators:
# →   GPU: NVIDIA_TESLA_T4, NVIDIA_A100_80GB, ...
# →   TPU: GOOGLE_TPU_V5E1, ...
```

## Portability

The folder is self-contained. To use it somewhere else:

```bash
cp -r colab-cli ~/wherever
cd ~/wherever
npm install
npx colab ls   # works immediately
```

- `.env` (OAuth client credentials) travels with the folder
- `.google_token.json` (auth session) travels with the folder — no need to re-auth
- Both are gitignored
- Token auto-refreshes using the stored refresh token

## How it works

1. **Auth**: Opens browser for Google OAuth (same flow as the Colab VS Code extension). Tokens cached locally at `.google_token.json` and auto-refreshed.
2. **Runtime management**: Talks to Colab's GAPI (`colab.pa.googleapis.com`) to create/list/connect to runtimes.
3. **Code execution**: Connects to the runtime's Jupyter kernel via WebSocket and sends `execute_request` messages. Streams stdout/stderr back in real-time.

## Requirements

- Node.js 18+
- Google Colab account (Pro/Pro+ recommended for GPU access)
- Colab VS Code extension installed (for OAuth credentials only — just needs to be installed, not running)
