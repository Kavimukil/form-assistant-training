#!/usr/bin/env bash
set -euo pipefail
echo "setup_remote.sh is retained for compatibility; using WSL2 bootstrap." >&2
exec "$(cd "$(dirname "$0")" && pwd)/bootstrap_wsl.sh" "$@"
