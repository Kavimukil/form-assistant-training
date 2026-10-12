#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/artifacts/functiongemma-phase0-remote.tar.gz}"
mkdir -p "$(dirname "$OUT")"
tar --exclude='.venv' --exclude='artifacts/checkpoints' --exclude='__pycache__' --exclude='.git' -czf "$OUT" -C "$ROOT" .
echo "$OUT"
