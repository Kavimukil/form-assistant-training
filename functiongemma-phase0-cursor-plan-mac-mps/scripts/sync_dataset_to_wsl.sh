#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 user@host [remote_linux_path] [ssh_port]"; exit 2; }
REMOTE="$1"; DEST="${2:-~/functiongemma-phase0}"; PORT="${3:-22}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/generated/phase0-v1"
[[ -d "$SRC" ]] || { echo "ERROR: $SRC does not exist. Pass G5/build the dataset first." >&2; exit 1; }
ssh -p "$PORT" "$REMOTE" "mkdir -p '$DEST/generated/phase0-v1'"
rsync -az --delete --human-readable --info=stats2,progress2 \
  -e "ssh -p $PORT" "$SRC/" "$REMOTE:$DEST/generated/phase0-v1/"
echo "Dataset synced incrementally to $REMOTE:$DEST/generated/phase0-v1"
