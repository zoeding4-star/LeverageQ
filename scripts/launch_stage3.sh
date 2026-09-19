#!/usr/bin/env bash
# One Stage-3 flow-region run on a currently idle GPU.
# Usage: bash launch_stage3.sh <idle_gpu> <mask> <seed> [smoke|full] [natural|norm]
set -euo pipefail

GPU="${1:?idle gpu index}"
MASK="${2:?am_mask_name, e.g. late50}"
SEED="${3:?seed}"
MODE="${4:-full}"
BUDGET="${5:-natural}"
ROOT=/mnt/zoe/projects/qam
PY=/mnt/zoe/conda-envs/qam/bin/python
LOGDIR="$ROOT/exp/logs"
mkdir -p "$LOGDIR"
SAFE_MASK="${MASK//_/-}"
LOG="$LOGDIR/stage3-${SAFE_MASK}-s${SEED}-gpu${GPU}-${MODE}-${BUDGET}.log"

if [ "$BUDGET" = "norm" ] || [ "$BUDGET" = "equal" ]; then
  NORM=True
  GROUP_SUFFIX="-norm"
else
  NORM=False
  GROUP_SUFFIX=""
fi

export GPU
export HOME=/mnt/zoe/home
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export WANDB_DIR=/mnt/zoe/home/.wandb
export WANDB_ENTITY="${WANDB_ENTITY:-nightingale-314-}"
export WANDB_PROJECT="${WANDB_PROJECT:-qam-reproduce}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"

COMMON=(
  "$PY" main_stage3.py
  --agent=agents/qam_region.py
  --env_name=cube-double-play-singletask-task2-v0
  --sparse=False
  --horizon_length=5
  --agent.action_chunking=True
  --agent.inv_temp=1.0
  --agent.fql_alpha=0.0
  --agent.edit_scale=0.0
  --agent.am_mask_name="$MASK"
  --agent.am_budget_normalize="$NORM"
  --seed="$SEED"
)

echo "[$(date -Is)] STAGE3 GPU=$GPU mask=$MASK seed=$SEED mode=$MODE budget=$BUDGET entity=$WANDB_ENTITY" | tee -a "$LOG"

if [ "$MODE" = "smoke" ]; then
  bash "$ROOT/scripts/run_on_idle_gpu.sh" "${COMMON[@]}" \
    --run_group="stage3-region-smoke${GROUP_SUFFIX}" \
    --tags="STAGE3,${MASK},smoke,cube-double" \
    --offline_steps=200 \
    --online_steps=0 \
    --eval_interval=200 \
    --eval_episodes=2 \
    --log_interval=50 \
    --dump_interval=50 \
    --dump_samples=16 \
    --save_interval=0 \
    2>&1 | tee -a "$LOG"
  exit ${PIPESTATUS[0]}
fi

if [ "$MODE" != "full" ]; then
  echo "MODE must be smoke or full, got: $MODE" >&2
  exit 2
fi

bash "$ROOT/scripts/run_on_idle_gpu.sh" "${COMMON[@]}" \
  --run_group="stage3-region${GROUP_SUFFIX}" \
  --tags="STAGE3,${MASK},cube-double,screening" \
  --offline_steps=1000000 \
  --online_steps=500000 \
  --eval_interval=50000 \
  --eval_episodes=50 \
  --log_interval=5000 \
  --dump_interval=20000 \
  --dump_samples=32 \
  --save_interval=50000 \
  2>&1 | tee -a "$LOG"
exit ${PIPESTATUS[0]}
