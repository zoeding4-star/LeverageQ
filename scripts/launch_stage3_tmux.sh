#!/usr/bin/env bash
# Start Stage 3 on allowed idle GPUs (default 1,4,5) in tmux.
# Smoke first (all 9 masks, seed 10001), then full screening if smoke exits 0.
# Usage:
#   bash scripts/launch_stage3_tmux.sh          # smoke then full (detached master)
#   bash scripts/launch_stage3_tmux.sh smoke    # smoke only
#   bash scripts/launch_stage3_tmux.sh full     # full only (after smoke passed)
set -euo pipefail

PHASE="${1:-all}"
ROOT=/mnt/zoe/projects/qam
GPUS=(1 4 5)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$ROOT"

idle_ok() {
  local gpu="$1"
  local used procs
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  procs=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]' || true)
  [ "${procs:-0}" -eq 0 ] && [ "${used:-0}" -lt 200 ]
}

echo "[$(date -Is)] checking allowed GPUs ${GPUS[*]}"
for g in "${GPUS[@]}"; do
  if ! idle_ok "$g"; then
    echo "GPU $g is not idle. Aborting rather than sharing/stealing." >&2
    bash /mnt/zoe/scripts/list_idle_gpus.sh >&2 || true
    exit 1
  fi
done

# Keep the smoke→full sequencer in its own tmux so the caller can detach.
if [ "$PHASE" = "all" ] && [ "${STAGE3_IN_TMUX:-}" != "1" ]; then
  if tmux has-session -t qam-s3-master 2>/dev/null; then
    echo "tmux session qam-s3-master already exists; not clobbering it." >&2
    exit 1
  fi
  tmux new-session -d -s qam-s3-master \
    "export STAGE3_IN_TMUX=1; cd $ROOT && bash $ROOT/scripts/launch_stage3_tmux.sh all; echo EXIT:\$?; exec bash"
  echo "started tmux qam-s3-master (attach: tmux attach -t qam-s3-master)"
  echo "smoke workers: qam-s3-smoke-g1 / g4 / g5"
  echo "full workers after smoke: qam-s3-g1 / g4 / g5"
  exit 0
fi

start_workers() {
  local mode="$1"
  local prefix="$2"
  local i=0
  for g in "${GPUS[@]}"; do
    local sess="${prefix}-g${g}"
    if tmux has-session -t "$sess" 2>/dev/null; then
      echo "tmux session $sess already exists; not clobbering it." >&2
      exit 1
    fi
    tmux new-session -d -s "$sess" \
      "cd $ROOT && bash $ROOT/scripts/launch_stage3_worker.sh $g $i 3 $mode; echo EXIT:\$?; exec bash"
    echo "started tmux $sess  (attach: tmux attach -t $sess)"
    i=$((i + 1))
  done
}

wait_workers() {
  local mode="$1"
  echo "[$(date -Is)] waiting for $mode workers"
  while true; do
    local alive=0
    local i=0
    for g in "${GPUS[@]}"; do
      local log="$ROOT/exp/logs/stage3-worker-gpu${g}-shard${i}-${mode}.log"
      if [ ! -f "$log" ] || ! grep -q "worker GPU $g done" "$log"; then
        alive=1
      fi
      i=$((i + 1))
    done
    if [ "$alive" -eq 0 ]; then
      echo "[$(date -Is)] all $mode workers reported done"
      return 0
    fi
    sleep 30
  done
}

if [ "$PHASE" = "smoke" ] || [ "$PHASE" = "all" ]; then
  start_workers smoke qam-s3-smoke
fi

if [ "$PHASE" = "all" ]; then
  wait_workers smoke
  if grep -q "done fail=1" "$ROOT"/exp/logs/stage3-worker-gpu*-shard*-smoke.log; then
    echo "A smoke worker reported fail=1. Not launching full runs." >&2
    exit 1
  fi
  /mnt/zoe/conda-envs/qam/bin/python "$ROOT/scripts/check_stage3_smoke.py"
  start_workers full qam-s3
  echo "[$(date -Is)] full workers started"
elif [ "$PHASE" = "full" ]; then
  start_workers full qam-s3
elif [ "$PHASE" = "smoke" ]; then
  echo "smoke workers started. Attach with: tmux attach -t qam-s3-smoke-g1"
else
  echo "phase must be all|smoke|full" >&2
  exit 2
fi
