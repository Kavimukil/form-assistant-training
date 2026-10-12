#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'TXT'
Usage: scripts/sync_to_wsl.sh user@windows-or-wsl-host [remote_linux_path] [ssh_port]

Incrementally sync the Mac working tree to Ubuntu/WSL2 over SSH using .rsync-filter.
The remote destination MUST be inside the WSL Linux filesystem, e.g.
  /home/mukil/functiongemma-phase0
not /mnt/c/... .

Environment:
  DRY_RUN=1       show what would change without copying
  DELETE=0        do not delete remote files missing locally (default DELETE=1)
TXT
}

[[ $# -ge 1 ]] || { usage; exit 2; }
REMOTE="$1"
DEST="${2:-~/functiongemma-phase0}"
PORT="${3:-22}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

[[ -f "$ROOT/.rsync-filter" ]] || { echo "ERROR: missing $ROOT/.rsync-filter" >&2; exit 1; }

RSYNC_ARGS=(-az --human-readable --info=stats2,progress2 --filter="merge $ROOT/.rsync-filter")
[[ "${DELETE:-1}" == "1" ]] && RSYNC_ARGS+=(--delete --delete-excluded)
[[ "${DRY_RUN:-0}" == "1" ]] && RSYNC_ARGS+=(--dry-run --itemize-changes)

ssh -p "$PORT" "$REMOTE" "mkdir -p '$DEST'"
rsync "${RSYNC_ARGS[@]}" -e "ssh -p $PORT" "$ROOT/" "$REMOTE:$DEST/"

echo "Synced Mac -> WSL2: $REMOTE:$DEST"
echo "Next on WSL2: cd '$DEST' && ./scripts/bootstrap_wsl.sh"
