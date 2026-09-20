#!/usr/bin/env bash
# One Stage-3B floor+boost run on a currently idle GPU.
# Usage: bash launch_stage3b.sh <idle_gpu> <mask> <seed> [smoke|full]
set -euo pipefail

GPU="${1:?idle gpu index}"
MASK="${2:?am_mask_name, e.g. late_b1}"
SEED="${3:?seed}"
MODE="${4:-full}"
ROOT=/mnt/zoe/projects/qam
PY=/mnt/zoe/conda-envs/qam/bin/python
LOGDIR="$ROOT/exp/logs"
mkdir -p "$LOGDIR"
SAFE_MASK="${MASK//_/-}"
LOG="$LOGDIR/stage3b-${SAFE_MASK}-s${SEED}-gpu${GPU}-${MODE}.log"

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

# Paper QAM on cube-double (Table 4 + Table 5 + reproduce.py):
# inv_temp=1, fql_alpha=0, edit_scale=0, h=5 chunking, K=10, rho=0.5, T=10,
# 1M offline + 500k online. Task 4 is the second official tuning task.
COMMON=(
  "$PY" main_stage3.py
  --agent=agents/qam_region.py
  --env_name=cube-double-play-singletask-task4-v0
  --sparse=False
  --horizon_length=5
  --agent.action_chunking=True
  --agent.inv_temp=1.0
  --agent.fql_alpha=0.0
  --agent.edit_scale=0.0
  --agent.num_qs=10
  --agent.rho=0.5
  --agent.flow_steps=10
  --agent.clip_grad=True
  --agent.am_mask_name="$MASK"
  --agent.am_budget_normalize=False
  --seed="$SEED"
)

echo "[$(date -Is)] STAGE3B GPU=$GPU mask=$MASK seed=$SEED mode=$MODE env=c2-task4 inv_temp=1 entity=$WANDB_ENTITY" | tee -a "$LOG"

ALLOWED_SEEDS="${STAGE3_SEEDS:-10001}"
seed_ok=0
for s in $ALLOWED_SEEDS; do
  if [ "$SEED" = "$s" ]; then
    seed_ok=1
    break
  fi
done
if [ "$MODE" = "full" ] && [ "${STAGE3_ALL_SEEDS:-0}" != "1" ] && [ "$seed_ok" -eq 0 ]; then
  echo "[$(date -Is)] SKIP mask=$MASK seed=$SEED (1-seed debug only; set STAGE3_ALL_SEEDS=1 to run extra seeds)" | tee -a "$LOG"
  exit 0
fi

if [ "$MODE" = "smoke" ]; then
  bash "$ROOT/scripts/run_on_idle_gpu.sh" "${COMMON[@]}" \
    --run_group=stage3b-floor-boost-smoke \
    --tags="STAGE3B,${MASK},smoke,cube-double-task4" \
    --offline_steps=200 \
    --online_steps=0 \
    --eval_interval=200 \
    --eval_episodes=2 \
    --log_interval=50 \
    --diag_interval=0 \
    --dump_interval=0 \
    --save_interval=0 \
    2>&1 | tee -a "$LOG"
  exit ${PIPESTATUS[0]}
fi

if [ "$MODE" != "full" ]; then
  echo "MODE must be smoke or full, got: $MODE" >&2
  exit 2
fi

bash "$ROOT/scripts/run_on_idle_gpu.sh" "${COMMON[@]}" \
  --run_group=stage3b-floor-boost \
  --tags="STAGE3B,${MASK},cube-double-task4,screening" \
  --offline_steps=1000000 \
  --online_steps=500000 \
  --eval_interval=50000 \
  --eval_episodes=50 \
  --log_interval=5000 \
  --diag_interval=0 \
  --dump_interval=0 \
  --env_log_interval=5000 \
  --save_interval=50000 \
  2>&1 | tee -a "$LOG"
exit ${PIPESTATUS[0]}
