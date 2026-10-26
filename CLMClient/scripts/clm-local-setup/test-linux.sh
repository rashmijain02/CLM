#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash -n "$SCRIPT_DIR/setup-linux.sh"
bash -n "$SCRIPT_DIR/server.sh"
bash "$SCRIPT_DIR/setup-linux.sh" --help >/dev/null

set +e
bash "$SCRIPT_DIR/server.sh" >/dev/null 2>&1
status=$?
set -e
if [[ $status -ne 2 ]]; then
    echo "server.sh without a command returned $status; expected 2." >&2
    exit 1
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[[ -f "$REPO_ROOT/pyproject.toml" ]] || {
    echo "Repository path resolution failed: $REPO_ROOT" >&2
    exit 1
}

echo "Standalone Linux setup script tests passed."

