#!/usr/bin/env bash
# QAM cube-double-task2 with flow-time raw-signal logging (qam_diag / main_diag).
# Official algorithm unchanged; this process only records extra tensors.
# Usage: bash launch_cube_double_diag.sh <idle_gpu> <seed> [smoke|full]
set -euo pipefail

GPU="${1:?idle gpu index}"
SEED="${2:?seed}"
MODE="${3:-full}"
ROOT=/mnt/zoe/projects/qam
PY=/mnt/zoe/conda-envs/qam/bin/python
LOGDIR="$ROOT/exp/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/cube-double-diag-s${SEED}-gpu${GPU}-${MODE}.log"

export GPU
export HOME=/mnt/zoe/home
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export WANDB_DIR=/mnt/zoe/home/.wandb

cd "$ROOT"

COMMON=(
  "$PY" main_diag.py
  --agent=agents/qam_diag.py
  --env_name=cube-double-play-singletask-task2-v0
  --sparse=False
  --horizon_length=5
  --agent.action_chunking=True
  --agent.inv_temp=1.0
  --agent.fql_alpha=0.0
  --agent.edit_scale=0.0
  --seed="$SEED"
)

if [ "$MODE" = "smoke" ]; then
  echo "[$(date -Is)] GPU=$GPU seed=$SEED DIAG SMOKE" | tee -a "$LOG"
  bash "$ROOT/scripts/run_on_idle_gpu.sh" "${COMMON[@]}" \
    --run_group=reproduce-double-diag-smoke \
    --tags=QAM,cube-double,diag,smoke \
    --offline_steps=200 \
    --online_steps=0 \
    --eval_interval=200 \
    --eval_episodes=2 \
    --log_interval=50 \
    --dump_interval=50 \
    --dump_samples=64 \
    --save_interval=0 \
    2>&1 | tee -a "$LOG"
  exit ${PIPESTATUS[0]}
fi

if [ "$MODE" != "full" ]; then
  echo "MODE must be smoke or full, got: $MODE" >&2
  exit 2
fi

echo "[$(date -Is)] GPU=$GPU seed=$SEED DIAG FULL" | tee -a "$LOG"
bash "$ROOT/scripts/run_on_idle_gpu.sh" "${COMMON[@]}" \
  --run_group=reproduce-double-diag \
  --tags=QAM,cube-double,diag \
  --offline_steps=1000000 \
  --online_steps=500000 \
  --eval_interval=50000 \
  --eval_episodes=50 \
  --log_interval=5000 \
  --dump_interval=10000 \
  --dump_samples=64 \
  --save_interval=50000 \
  2>&1 | tee -a "$LOG"
exit ${PIPESTATUS[0]}
