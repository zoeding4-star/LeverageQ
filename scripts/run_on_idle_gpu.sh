#!/usr/bin/env bash
# Run a command on one currently idle GPU. Refuse if that GPU is occupied.
# Usage:
#   GPU=1 bash /mnt/zoe/projects/qam/scripts/run_on_idle_gpu.sh python main.py ...
set -euo pipefail

if [ -z "${GPU:-}" ]; then
  echo "Set GPU to an idle index first, e.g. GPU=1" >&2
  echo "Check idle cards with: bash /mnt/zoe/scripts/list_idle_gpus.sh" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
procs=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name --format=csv,noheader)
if [ -n "$procs" ] || [ "${used:-0}" -ge 200 ]; then
  echo "Refusing to run: GPU $GPU is occupied (used=${used} MiB)." >&2
  echo "$procs" >&2
  echo "This would interrupt someone else's job. Pick an IDLE GPU." >&2
  bash /mnt/zoe/scripts/list_idle_gpus.sh >&2 || true
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Do not let JAX see or initialize other GPUs.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "[guard] GPU=$GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES used=${used}MiB"
cd /mnt/zoe/projects/qam
exec "$@"
