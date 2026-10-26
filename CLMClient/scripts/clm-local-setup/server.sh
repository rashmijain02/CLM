#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CLIENT_ROOT="$REPO_ROOT/CLMClient"
VENV="${CLM_VENV:-$REPO_ROOT/.venv-real}"
ENCODER_URL="${CLM_EMB_URL:-http://127.0.0.1:8090/v1/embeddings}"
API_URL="${CLM_BASE_URL:-http://127.0.0.1:8700}"

require_environment() {
    [[ -x "$VENV/bin/python" ]] || {
        echo "Missing $VENV. Run setup-linux.sh first." >&2
        exit 1
    }
}

encoder_health() {
    local models_url="${ENCODER_URL%/v1/embeddings}/v1/models"
    curl --fail --silent --show-error "$models_url"
    echo
}

api_health() {
    local body
    body="$(curl --fail --silent --show-error "$API_URL/health")"
    HEALTH_JSON="$body" "$VENV/bin/python" - <<'PY'
import json
import os

health = json.loads(os.environ["HEALTH_JSON"])
if health.get("mock"):
    raise SystemExit("The endpoint is the mock server, not the real CLM server.")
if not health.get("ok"):
    raise SystemExit("CLM API reports ok=false.")
if not health.get("embedder"):
    raise SystemExit("CLM API is running but its embedding server is unavailable.")
print(json.dumps(health, indent=2))
PY
}

case "${1:-}" in
    encoder)
        require_environment
        cd "$REPO_ROOT"
        export PATH="$VENV/bin:$PATH"
        export GPU="${GPU:-0}"
        export PORT="${CLM_EMB_PORT:-8090}"
        export UTIL="${CLM_GPU_UTIL:-0.75}"
        export MAXLEN="${CLM_MAX_TOKENS:-2048}"
        exec "$REPO_ROOT/serve_qwen3_8b.sh"
        ;;
    encoder-health)
        encoder_health
        ;;
    api)
        require_environment
        cd "$REPO_ROOT"
        exec "$VENV/bin/clm-serve" \
            --emb-url "$ENCODER_URL" \
            --port "${CLM_PORT:-8700}" \
            --max-tokens "${CLM_MAX_TOKENS:-2048}"
        ;;
    health)
        require_environment
        encoder_health >/dev/null
        api_health
        ;;
    verify)
        require_environment
        encoder_health >/dev/null
        api_health >/dev/null
        cd "$CLIENT_ROOT"
        "$VENV/bin/python" main.py --scenario all
        "$VENV/bin/python" main.py --scenario rank
        echo "Real CLM server and client verification passed."
        ;;
    *)
        echo "Usage: server.sh {encoder|encoder-health|api|health|verify}" >&2
        exit 2
        ;;
esac
