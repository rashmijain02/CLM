# Standalone CLM setup scripts

These scripts install, start, test, and verify CLM without requiring an agent or
skill runtime. Run them from any working directory; each script resolves the
repository from its own location.

## Windows client

Preflight:

```powershell
.\setup-windows.ps1 -Mode Client -CheckOnly
```

Install:

```powershell
.\setup-windows.ps1 -Mode Client
```

## Windows client and WSL2 server

Preflight:

```powershell
.\setup-windows.ps1 -Mode WSL -Distribution Ubuntu-24.04 -CheckOnly
```

Install:

```powershell
.\setup-windows.ps1 -Mode WSL -Distribution Ubuntu-24.04
```

If WSL or Ubuntu requires a restart or initial user creation, complete that
step and rerun the same command.

## Native Linux or WSL server

Preflight:

```bash
bash setup-linux.sh --check-only
```

Install:

```bash
bash setup-linux.sh
```

Start the encoder and API in separate terminals:

```bash
bash server.sh encoder
bash server.sh encoder-health
bash server.sh api
bash server.sh health
bash server.sh verify
```

## Independent script tests

Windows:

```powershell
.\test-windows.ps1
```

Linux, WSL, or Git Bash:

```bash
bash test-linux.sh
```

These tests are non-destructive. They validate parsing, command interfaces, and
path resolution. The Windows test also runs the installed client preflight.

## Environment overrides

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLM_VENV` | `<repository>/.venv-real` | Linux server virtual environment |
| `CLM_EMB_URL` | `http://127.0.0.1:8090/v1/embeddings` | Encoder endpoint |
| `CLM_BASE_URL` | `http://127.0.0.1:8700` | CLM API endpoint |
| `CLM_GPU_UTIL` | `0.75` | vLLM GPU-memory fraction |
| `CLM_MAX_TOKENS` | `2048` | Encoder and CLM token limit |
| `CLM_EMB_PORT` | `8090` | Local encoder port |
| `CLM_PORT` | `8700` | Local CLM API port |

