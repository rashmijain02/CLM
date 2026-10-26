# CLM local setup guide

## Choose a setup

| Goal | Environment | GPU | Result |
| --- | --- | --- | --- |
| Learn and test `CLMClient` | Native Windows | Not required | Mock API; predictions are not meaningful |
| Run real CLM inference | WSL2 Ubuntu or Linux | NVIDIA, about 16 GB+ VRAM | Qwen3-8B embeddings and trained CLM heads |
| Call an existing server | Any Python 3.10+ system | Not required locally | Real predictions from the remote server |

vLLM does not support native Windows. The Windows setup therefore installs the
client package with `--no-deps` and installs only the lightweight dependencies
needed by `CLMClient` and the mock server.

## Automated setup

The recommended setup interface is the
standalone [`scripts/clm-local-setup`](scripts/clm-local-setup/README.md)
toolkit. The optional
[`clm-local-setup`](.github/skills/clm-local-setup/SKILL.md) skill tells an agent
how to orchestrate the same scripts, but is not required.

From `CLMClient` on Windows:

```powershell
# Client only
.\scripts\clm-local-setup\setup-windows.ps1 -Mode Client

# Real server in WSL2
.\scripts\clm-local-setup\setup-windows.ps1 -Mode WSL -Distribution Ubuntu-24.04
```

From `CLMClient` on Linux:

```bash
bash scripts/clm-local-setup/setup-linux.sh
```

The remaining sections explain the same process manually and provide
troubleshooting context.

## A. Native Windows client and mock server

### 1. Install Python

Open PowerShell:

```powershell
winget install --id Python.Python.3.12 --exact --scope user
```

Close and reopen PowerShell so the new Python launcher is available:

```powershell
py -3.12 --version
```

### 2. Create the environment

```powershell
cd H:\masterswork\gitrash\CLM\CLMClient
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

The script:

1. Creates `CLMClient\.venv`.
2. Installs the lightweight client/mock dependencies.
3. Installs the repository's `contrastive-lm` package in editable mode without
   attempting to install native-Windows vLLM.
4. Copies `.env.example` to `.env` when needed.

### 3. Start the mock API

```powershell
.\.venv\Scripts\python.exe ..\tools\playground_mock.py
```

Expected health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8700/health
```

The response includes `"mock": true`. Do not use mock scores to evaluate model
quality or make decisions.

### 4. Run the examples

In a second PowerShell window:

```powershell
cd H:\masterswork\gitrash\CLM\CLMClient
.\.venv\Scripts\python.exe .\main.py --scenario all
.\.venv\Scripts\python.exe .\main.py --scenario rank
```

## B. Real CLM server with WSL2 and NVIDIA

The following installation is separate from the lightweight Windows
environment.

### 1. Install Ubuntu in WSL2

Run an elevated PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows if requested, launch Ubuntu, and create the Linux user account.
Verify that the Windows NVIDIA driver is visible inside Ubuntu:

```bash
nvidia-smi
```

Do not install a separate Linux NVIDIA display driver inside WSL.

### 2. Install Python tooling

Inside Ubuntu:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git curl
cd /mnt/h/masterswork/gitrash/CLM
python3 -m venv .venv-real
source .venv-real/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs PyTorch, vLLM, FastAPI, the editable CLM package, and the remaining
runtime dependencies. The first installation can take several minutes.

### 3. Start the Qwen3-8B embedding server

In the first Ubuntu terminal:

```bash
cd /mnt/h/masterswork/gitrash/CLM
source .venv-real/bin/activate
GPU=0 PORT=8090 UTIL=0.70 MAXLEN=2048 ./serve_qwen3_8b.sh
```

The initial run downloads Qwen3-8B and requires substantial disk space. Wait
until the embedding endpoint is ready:

```bash
curl http://127.0.0.1:8090/v1/models
```

If the process runs out of GPU memory, lower `UTIL`; if startup fails because
the model cannot reserve enough memory, stop other GPU workloads first.

### 4. Start the CLM API

In a second Ubuntu terminal:

```bash
cd /mnt/h/masterswork/gitrash/CLM
source .venv-real/bin/activate
clm-serve --emb-url http://127.0.0.1:8090/v1/embeddings --port 8700
```

The first run downloads the released projection-head checkpoint. Verify:

```bash
curl http://127.0.0.1:8700/health
```

### 5. Call the real server

The Windows example can call the WSL service through `localhost`:

```powershell
cd H:\masterswork\gitrash\CLM\CLMClient
.\.venv\Scripts\python.exe .\main.py --scenario all
```

If `localhost` forwarding is disabled, obtain the WSL address with
`wsl hostname -I` and set `CLM_BASE_URL` in `.env`.

## C. Existing or remote CLM server

Edit `.env`:

```dotenv
CLM_BASE_URL=https://your-clm-host.example
CLM_MODEL=clm-latest
CLM_API_KEY=replace-me
```

Then run `main.py` normally. Keep `.env` private; it is excluded from Git.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLM_BASE_URL` | `http://127.0.0.1:8700` | CLM API root |
| `CLM_MODEL` | `clm-latest` | Served checkpoint name |
| `CLM_API_KEY` | unset | Optional server authentication |

Command-line `--base-url` and `--model` values override the corresponding
environment variables.

## Troubleshooting

### `Permission denied (publickey)` while cloning

The repository clone used an SSH URL without a GitHub-authorized SSH key. Use
an HTTPS clone URL or add the public key to the GitHub account that can access
the repository.

### `Python was not found`

Install Python with `winget`, close every PowerShell window, and open a new one.
The Microsoft Store application alias is not a working Python installation.

### `No module named clm`

Run `setup.ps1` from `CLMClient`, then use
`CLMClient\.venv\Scripts\python.exe` rather than a global Python executable.

### The example says the server is unhealthy

Start either `tools/playground_mock.py` or the real `clm-serve` process. Confirm
that <http://127.0.0.1:8700/health> responds.

### The mock gives unexpected answers

This is expected. It uses character n-gram overlap to exercise the real API
schema and is not the trained CLM model.
