#!/usr/bin/env bash
# Run one target experiment and leave a machine-readable completion marker.
set -uo pipefail

GPU="${1:?gpu}"
MASK="${2:?mask}"
MODE="${3:?mode}"
DONE="${4:?done marker}"
ROOT=/mnt/zoe/projects/qam

cd "$ROOT"
bash scripts/launch_target_screen.sh "$GPU" "$MASK" "$MODE"
status=$?
mkdir -p "$(dirname "$DONE")"
printf '%s\n' "$status" > "$DONE"
exit "$status"
