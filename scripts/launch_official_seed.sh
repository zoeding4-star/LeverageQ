#!/usr/bin/env bash
# Launch one official plain-QAM cube-triple-task2 seed on an idle GPU.
# Usage: bash launch_official_seed.sh <gpu> <seed>
set -euo pipefail

GPU="${1:?gpu index}"
SEED="${2:?seed}"
ROOT=/mnt/zoe/projects/qam
PY=/mnt/zoe/conda-envs/qam/bin/python

export GPU
export HOME=/mnt/zoe/home
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

cd "$ROOT"
exec bash "$ROOT/scripts/run_on_idle_gpu.sh" "$PY" main.py \
  --run_group=reproduce \
  --agent=agents/qam.py \
  --tags=QAM \
  --seed="$SEED" \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --sparse=False \
  --horizon_length=5 \
  --agent.action_chunking=True \
  --agent.inv_temp=3.0 \
  --agent.fql_alpha=0.0 \
  --agent.edit_scale=0.0
