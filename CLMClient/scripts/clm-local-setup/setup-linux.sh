#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="${CLM_VENV:-$REPO_ROOT/.venv-real}"
CHECK_ONLY=0
SKIP_SYSTEM_PACKAGES=0

usage() {
    cat <<'EOF'
Usage: setup-linux.sh [--check-only] [--skip-system-packages] [--venv PATH]

Install and validate the real CLM server environment on Linux or WSL2.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        --skip-system-packages)
            SKIP_SYSTEM_PACKAGES=1
            shift
            ;;
        --venv)
            [[ $# -ge 2 ]] || { echo "--venv requires a path" >&2; exit 2; }
            VENV="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "$(uname -s)" == "Linux" ]] || {
    echo "Real CLM setup requires Linux or WSL2; found $(uname -s)." >&2
    exit 1
}
[[ -f "$REPO_ROOT/pyproject.toml" ]] || {
    echo "Could not locate the CLM repository at $REPO_ROOT." >&2
    exit 1
}

echo "CLM repository: $REPO_ROOT"
echo "Virtual environment: $VENV"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU:"
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader
else
    echo "nvidia-smi is unavailable. Real Qwen3-8B/vLLM inference requires an NVIDIA GPU." >&2
    exit 1
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    command -v python3 >/dev/null 2>&1 || {
        echo "python3 is missing." >&2
        exit 1
    }
    python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
    command -v curl >/dev/null 2>&1 || {
        echo "curl is missing." >&2
        exit 1
    }
    if [[ -x "$VENV/bin/python" ]]; then
        "$VENV/bin/python" -c 'import clm, torch, vllm; print("Installed CLM environment: OK")'
    else
        echo "CLM environment is not installed yet at $VENV." >&2
        exit 1
    fi
    exit 0
fi

if [[ $SKIP_SYSTEM_PACKAGES -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    SUDO=()
    if [[ $EUID -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 || {
            echo "sudo is required to install system packages." >&2
            exit 1
        }
        SUDO=(sudo)
    fi
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y python3-venv python3-pip git curl
fi

command -v python3 >/dev/null 2>&1 || {
    echo "python3 is missing; install Python 3.10 or newer." >&2
    exit 1
}
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
command -v curl >/dev/null 2>&1 || {
    echo "curl is missing; install it or rerun without --skip-system-packages." >&2
    exit 1
}

cd "$REPO_ROOT"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$REPO_ROOT/requirements.txt"
"$VENV/bin/python" -c \
    'import clm, torch, vllm; print(f"CLM {clm.__version__}; torch CUDA available: {torch.cuda.is_available()}")'
"$VENV/bin/clm-download"

echo
echo "Real CLM server dependencies are installed."
echo "Start the encoder: bash $SCRIPT_DIR/server.sh encoder"
echo "Then start the API: bash $SCRIPT_DIR/server.sh api"
