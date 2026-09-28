#!/usr/bin/env bash
# One target-weight QAM run. GPU must already be idle.
set -euo pipefail

GPU="${1:?gpu index}"
MASK="${2:?target mask}"
MODE="${3:-full}"
ROOT=/mnt/zoe/projects/qam
PY=/mnt/zoe/conda-envs/qam/bin/python
SEED=10001
LOGDIR="$ROOT/exp/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/target-${MASK}-s${SEED}-gpu${GPU}-${MODE}.log"

case "$GPU" in
  0|1|2|3|4|5|6|7) ;;
  *) echo "GPU $GPU is outside the machine GPU set 0..7" >&2; exit 2 ;;
esac

export GPU
export HOME=/mnt/zoe/home
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export WANDB_DIR=/mnt/zoe/home/.wandb
export WANDB_ENTITY="${WANDB_ENTITY:-nightingale-314-}"
export WANDB_PROJECT=qam-target-region
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

COMMON=(
  "$PY" main_target_region.py
  --agent=agents/qam_target_region.py
  --env_name=cube-double-play-singletask-task2-v0
  --sparse=False
  --horizon_length=5
  --agent.action_chunking=True
  --agent.inv_temp=1.0
  --agent.fql_alpha=0.0
  --agent.edit_scale=0.0
  --agent.residual=False
  --agent.target_mask_name="$MASK"
  --seed="$SEED"
  --wandb_project=qam-target-region
)

echo "[$(date -Is)] target-weight GPU=$GPU mask=$MASK mode=$MODE" | tee -a "$LOG"
if [ "$MODE" = smoke ]; then
  RUN_NAME="c2t2-target-${MASK}-s${SEED}-smoke"
  bash "$ROOT/scripts/run_on_idle_gpu.sh" "${COMMON[@]}" \
    --run_group=smoke \
    --wandb_run_name="$RUN_NAME" \
    --tags="TARGET_WEIGHT,${MASK},smoke,cube-double-task2" \
    --offline_steps=200 --online_steps=0 \
    --eval_interval=200 --eval_episodes=2 \
    --log_interval=50 --diag_interval=50 \
    --dump_interval=0 --save_interval=0 \
    2>&1 | tee -a "$LOG"
  exit "${PIPESTATUS[0]}"
fi

if [ "$MODE" != full ]; then
  echo "mode must be smoke or full" >&2
  exit 2
fi

RUN_NAME="c2t2-target-${MASK}-s${SEED}-400k"
bash "$ROOT/scripts/run_on_idle_gpu.sh" "${COMMON[@]}" \
  --run_group=screen-400k \
  --wandb_run_name="$RUN_NAME" \
  --tags="TARGET_WEIGHT,${MASK},screen,cube-double-task2" \
  --offline_steps=400000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=100 \
  --log_interval=5000 --diag_interval=20000 \
  --dump_interval=0 --env_log_interval=5000 \
  --save_interval=0 --auto_cleanup=True \
  2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
