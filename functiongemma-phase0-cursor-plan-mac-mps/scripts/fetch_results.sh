#!/usr/bin/env bash
set -euo pipefail
echo "fetch_results.sh now pulls lightweight WSL2 results; checkpoints require pull_checkpoints_from_wsl.sh." >&2
exec "$(cd "$(dirname "$0")" && pwd)/pull_results_from_wsl.sh" "$@"
