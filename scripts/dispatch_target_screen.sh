#!/usr/bin/env bash
# Poll all eight GPUs and dispatch each job to any card that becomes truly idle.
set -euo pipefail

ROOT=/mnt/zoe/projects/qam
STATE="$ROOT/exp/target_scheduler"
MASKS=(all early50 late50 noq late25 remove_late25)
GPUS=(0 1 2 3 4 5 6 7)
mkdir -p "$STATE"

idle_ok() {
  local gpu="$1" used procs
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  procs=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]' || true)
  [ "${procs:-0}" -eq 0 ] && [ "${used:-0}" -lt 200 ]
}

run_phase() {
  local phase="$1"
  declare -A assigned=()
  echo "[$(date -Is)] dynamic phase=$phase waiting on GPUs ${GPUS[*]}"
  while true; do
    local complete=0 failed=0
    declare -A reserved=()

    for mask in "${MASKS[@]}"; do
      local done="$STATE/${phase}-${mask}.done"
      local sess="qam-target-${phase}-${mask}"
      if [ -f "$done" ]; then
        status=$(tr -d '[:space:]' < "$done")
        if [ "$status" = 0 ]; then
          complete=$((complete + 1))
        else
          echo "[$(date -Is)] $phase/$mask failed status=$status" >&2
          failed=1
        fi
      elif tmux has-session -t "$sess" 2>/dev/null; then
        if [ -n "${assigned[$mask]:-}" ]; then
          reserved[${assigned[$mask]}]=1
        fi
      fi
    done

    [ "$failed" -eq 0 ] || return 1
    if [ "$complete" -eq "${#MASKS[@]}" ]; then
      echo "[$(date -Is)] dynamic phase=$phase complete"
      return 0
    fi

    for mask in "${MASKS[@]}"; do
      local done="$STATE/${phase}-${mask}.done"
      local sess="qam-target-${phase}-${mask}"
      [ -f "$done" ] && continue
      tmux has-session -t "$sess" 2>/dev/null && continue
      for gpu in "${GPUS[@]}"; do
        [ -n "${reserved[$gpu]:-}" ] && continue
        if idle_ok "$gpu"; then
          assigned[$mask]="$gpu"
          reserved[$gpu]=1
          tmux new-session -d -s "$sess" \
            "cd $ROOT && bash scripts/run_target_slot.sh $gpu $mask $phase $done"
          echo "[$(date -Is)] dispatched $phase/$mask to GPU $gpu session=$sess"
          break
        fi
      done
    done
    sleep 30
  done
}

run_phase smoke
run_phase full
echo "[$(date -Is)] all target-weight experiments complete"
