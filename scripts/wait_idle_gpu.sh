#!/usr/bin/env bash
# Wait until GPU $1 is idle (<200 MiB and no compute proc), then exit 0.
# Never touches other GPUs.
set -euo pipefail
GPU="${1:?gpu index}"
SLEEP="${2:-30}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
while true; do
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  procs=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]' || true)
  if [ "${procs:-0}" -eq 0 ] && [ "${used:-0}" -lt 200 ]; then
    echo "[$(date -Is)] GPU $GPU idle (used=${used}MiB)"
    exit 0
  fi
  echo "[$(date -Is)] GPU $GPU busy used=${used}MiB procs=${procs}; sleep ${SLEEP}s"
  sleep "$SLEEP"
done
