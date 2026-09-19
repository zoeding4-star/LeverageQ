#!/usr/bin/env bash
# Smoke then full diagnostic run on one idle GPU.
set -euo pipefail
GPU="${1:?idle gpu}"
SEED="${2:?seed}"
ROOT=/mnt/zoe/projects/qam
LOGDIR="$ROOT/exp/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/cube-double-diag-chain-s${SEED}-gpu${GPU}.log"
echo "[$(date -Is)] diag smoke GPU=$GPU seed=$SEED" | tee -a "$LOG"
if ! bash "$ROOT/scripts/launch_cube_double_diag.sh" "$GPU" "$SEED" smoke; then
  echo "[$(date -Is)] DIAG SMOKE FAILED" | tee -a "$LOG"
  exit 1
fi
echo "[$(date -Is)] diag smoke ok; starting full diag" | tee -a "$LOG"
exec bash "$ROOT/scripts/launch_cube_double_diag.sh" "$GPU" "$SEED" full
