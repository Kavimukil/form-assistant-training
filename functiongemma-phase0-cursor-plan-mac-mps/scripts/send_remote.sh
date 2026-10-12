#!/usr/bin/env bash
set -euo pipefail
echo "send_remote.sh now uses incremental Mac -> WSL2 rsync." >&2
exec "$(cd "$(dirname "$0")" && pwd)/sync_to_wsl.sh" "$@"
