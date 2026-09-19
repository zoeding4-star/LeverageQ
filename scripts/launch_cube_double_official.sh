#!/usr/bin/env bash
# Official QAM on cube-double-task2: short smoke, then full paper config if smoke exits 0.
# Usage: bash launch_cube_double_official.sh <idle_gpu> <seed>
# Does not edit agents/qam.py or main.py.
set -euo pipefail

GPU="${1:?idle gpu index}"
SEED="${2:?seed, official protocol starts at 10001}"
ROOT=/mnt/zoe/projects/qam
PY=/mnt/zoe/conda-envs/qam/bin/python
LOGDIR="$ROOT/exp/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/cube-double-official-s${SEED}-gpu${GPU}.log"

export GPU
export HOME=/mnt/zoe/home
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export WANDB_DIR=/mnt/zoe/home/.wandb

cd "$ROOT"

COMMON=(
  "$PY" main.py
  --agent=agents/qam.py
  --env_name=cube-double-play-singletask-task2-v0
  --sparse=False
  --horizon_length=5
  --agent.action_chunking=True
  --agent.inv_temp=1.0
  --agent.fql_alpha=0.0
  --agent.edit_scale=0.0
  --seed="$SEED"
)

echo "[$(date -Is)] GPU=$GPU seed=$SEED SMOKE start" | tee -a "$LOG"
if ! bash "$ROOT/scripts/run_on_idle_gpu.sh" "${COMMON[@]}" \
  --run_group=reproduce-double-smoke \
  --tags=QAM,cube-double,smoke \
  --offline_steps=200 \
  --online_steps=0 \
  --eval_interval=200 \
  --eval_episodes=4 \
  --log_interval=50 \
  --save_interval=0 \
  2>&1 | tee -a "$LOG"; then
  echo "[$(date -Is)] SMOKE FAILED; not launching full run" | tee -a "$LOG"
  exit 1
fi

echo "[$(date -Is)] SMOKE ok. Launching official 1M+500k on GPU $GPU" | tee -a "$LOG"
bash "$ROOT/scripts/run_on_idle_gpu.sh" "${COMMON[@]}" \
  --run_group=reproduce-double \
  --tags=QAM,cube-double \
  --offline_steps=1000000 \
  --online_steps=500000 \
  --eval_interval=50000 \
  --eval_episodes=50 \
  --log_interval=5000 \
  --save_interval=50000 \
  2>&1 | tee -a "$LOG"
exit ${PIPESTATUS[0]}
