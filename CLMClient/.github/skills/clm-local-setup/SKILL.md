---
name: clm-local-setup
description: Installs, starts, and verifies the CLM Python client and real Qwen3-8B CLM server on native Linux or Windows with WSL2. Use when setting up CLM locally, configuring vLLM, installing Ubuntu under WSL, starting the embedding and API services, troubleshooting GPU setup, or validating CLMClient scenarios.
license: Apache-2.0
compatibility: Requires internet access. Real inference requires Linux or WSL2, an NVIDIA GPU with about 16 GB or more VRAM, a working NVIDIA driver, Python 3.10+, and sufficient disk space for Qwen3-8B.
metadata:
  author: CLMClient
  version: "1.0.0"
---

# CLM local setup

Set up the repository with the standalone toolkit in
`../../../scripts/clm-local-setup/`. The skill contains orchestration
instructions only; the scripts remain independently usable and testable without
an agent. Do not replace them with ad hoc installation commands.

## 1. Detect the target

Determine the current operating environment before making changes:

- **Native Windows client only:** use PowerShell and
  `../../../scripts/clm-local-setup/setup-windows.ps1 -Mode Client`.
- **Windows with a real WSL server:** use PowerShell and
  `../../../scripts/clm-local-setup/setup-windows.ps1 -Mode WSL`.
- **Native Linux real server:** use Bash and
  `../../../scripts/clm-local-setup/setup-linux.sh`.
- **Existing remote server:** install the client on the local OS, set
  `CLM_BASE_URL`, and run the verification client.

Real vLLM inference is not supported on native Windows. Do not attempt to
install vLLM into the Windows virtual environment.

## 2. Run preflight first

Use the tool appropriate to the host:

```powershell
# Windows client
.\scripts\clm-local-setup\setup-windows.ps1 -Mode Client -CheckOnly

# Windows + WSL real server
.\scripts\clm-local-setup\setup-windows.ps1 -Mode WSL -Distribution Ubuntu-24.04 -CheckOnly
```

```bash
# Native Linux or from inside WSL
bash scripts/clm-local-setup/setup-linux.sh --check-only
```

Read and act on every failed check. Do not claim setup is complete after only
installing packages.

## 3. Install

### Native Windows client

Run with PowerShell:

```powershell
.\scripts\clm-local-setup\setup-windows.ps1 -Mode Client
```

This installs Python 3.12 with `winget` when needed, then invokes the
repository's client setup script. It intentionally installs the CLM package
without the Linux-only vLLM dependency.

### Windows with WSL2 real server

Run with PowerShell:

```powershell
.\scripts\clm-local-setup\setup-windows.ps1 -Mode WSL -Distribution Ubuntu-24.04
```

If Ubuntu is absent, the script starts `wsl --install`. A restart and initial
Ubuntu user creation may be required. After those steps, rerun the same command;
it delegates the actual Linux installation to `setup-linux.sh`.

### Native Linux real server

Run with Bash:

```bash
bash scripts/clm-local-setup/setup-linux.sh
```

The script verifies NVIDIA access, installs required OS packages on Debian or
Ubuntu, creates `.venv-real`, installs repository dependencies, validates
PyTorch/vLLM imports, and downloads the released CLM projection head.

## 4. Start the real services

Start each long-running service with the execution tool in background/detached
mode. Do not run both in a blocking terminal.

First start the encoder:

```bash
bash scripts/clm-local-setup/server.sh encoder
```

Wait until this succeeds:

```bash
bash scripts/clm-local-setup/server.sh encoder-health
```

Then start the CLM API in a second background process:

```bash
bash scripts/clm-local-setup/server.sh api
```

Do not continue until strict health succeeds:

```bash
bash scripts/clm-local-setup/server.sh health
```

`health` requires both `ok: true` and `embedder: true`, and rejects the mock
server. Merely receiving HTTP 200 is insufficient.

On a Windows host, invoke the same scripts through the selected distribution:

```powershell
wsl -d Ubuntu-24.04 -- bash <wsl-repository-path>/CLMClient/scripts/clm-local-setup/server.sh encoder
wsl -d Ubuntu-24.04 -- bash <wsl-repository-path>/CLMClient/scripts/clm-local-setup/server.sh api
```

Obtain `<wsl-repository-path>` with:

```powershell
wsl -d Ubuntu-24.04 -- wslpath -a <windows-repository-path>
```

## 5. Verify the client end to end

From Windows:

```powershell
.\scripts\clm-local-setup\verify-windows.ps1
```

From Linux or WSL:

```bash
bash scripts/clm-local-setup/server.sh verify
```

Verification must complete all of the following:

1. The embedding endpoint returns a model list.
2. `/health` reports `ok: true`.
3. `/health` reports `embedder: true`.
4. `/health` does not report `mock: true`.
5. All typed synthetic scenarios complete.
6. The ranking scenario completes.

Only then report that the real server and client are set up.

## 6. Handle common failures

- **No WSL distribution:** run the Windows WSL setup script and complete the
  required restart/user initialization before rerunning it.
- **`nvidia-smi` fails in WSL:** update the Windows NVIDIA driver. Do not install
  a Linux display driver inside WSL.
- **Encoder out of memory:** close other GPU workloads, then retry with a
  suitable `CLM_GPU_UTIL`, normally `0.70` to `0.85` on a 24 GB GPU.
- **Encoder health times out:** inspect the encoder process output; the first run
  downloads Qwen3-8B and can take a long time.
- **API health says `embedder: false`:** keep the API running only after the
  encoder is healthy and reachable on `CLM_EMB_URL`.
- **Windows cannot reach WSL localhost:** set `CLM_BASE_URL` to the WSL address
  returned by `wsl hostname -I`.

For detailed background and manual troubleshooting, read
`../../../SETUP_GUIDE.md`.
