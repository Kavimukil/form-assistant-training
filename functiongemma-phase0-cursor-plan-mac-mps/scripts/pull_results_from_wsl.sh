#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 user@host [remote_linux_path] [local_dir] [ssh_port]"; exit 2; }
REMOTE="$1"; SRC="${2:-~/functiongemma-phase0}"; LOCAL="${3:-artifacts/remote-results}"; PORT="${4:-22}"
mkdir -p "$LOCAL"
# Intentionally excludes checkpoint/model weight blobs. Use pull_checkpoints_from_wsl.sh explicitly.
rsync -az --human-readable --info=stats2,progress2 \
  --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' --exclude='*.pth' \
  --exclude='optimizer.pt' --exclude='scheduler.pt' \
  -e "ssh -p $PORT" \
  "$REMOTE:$SRC/outputs/" "$LOCAL/outputs/" || true
rsync -az --human-readable --info=stats2,progress2 \
  -e "ssh -p $PORT" "$REMOTE:$SRC/artifacts/gates/" "$LOCAL/gates/" || true
echo "Pulled lightweight results to $LOCAL (model/checkpoint weights intentionally excluded)."
