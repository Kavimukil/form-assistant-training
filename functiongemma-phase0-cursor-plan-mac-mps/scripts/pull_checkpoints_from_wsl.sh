#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 user@host [remote_linux_path] [local_dir] [ssh_port]"; exit 2; }
REMOTE="$1"; SRC="${2:-~/functiongemma-phase0}"; LOCAL="${3:-artifacts/remote-checkpoints}"; PORT="${4:-22}"
mkdir -p "$LOCAL"
# Explicit operation because checkpoints can be multi-GB. Partial transfers are resumable.
rsync -az --partial --append-verify --human-readable --info=stats2,progress2 \
  -e "ssh -p $PORT" "$REMOTE:$SRC/checkpoints/" "$LOCAL/"
echo "Pulled checkpoints to $LOCAL"
