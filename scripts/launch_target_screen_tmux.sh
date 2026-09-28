#!/usr/bin/env bash
# Launch target-weight workers in detached tmux sessions.
# "all" waits for allowed GPUs, runs smoke, then starts the full screen.
set -euo pipefail

MODE="${1:-all}"
shift || true
ROOT=/mnt/zoe/projects/qam
GPUS=("$@")

if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "usage: $0 all|smoke|full <allowed_gpu> [allowed_gpu ...]" >&2
  exit 2
fi

for gpu in "${GPUS[@]}"; do
  case "$gpu" in 1|4|5) ;; *) echo "GPU $gpu is not allowed" >&2; exit 2 ;; esac
done

start_workers() {
  local phase="$1" n=${#GPUS[@]}
  for i in "${!GPUS[@]}"; do
    local gpu=${GPUS[$i]} sess="qam-target-${phase}-g${gpu}"
    if tmux has-session -t "$sess" 2>/dev/null; then
      echo "tmux session $sess already exists" >&2
      exit 1
    fi
    tmux new-session -d -s "$sess" \
      "cd $ROOT && bash scripts/launch_target_screen_worker.sh $gpu $i $n $phase; echo EXIT:\$?; exec bash"
    echo "started $sess (tmux attach -t $sess)"
  done
}

wait_workers() {
  local phase="$1"
  while true; do
    local pending=0
    for i in "${!GPUS[@]}"; do
      local gpu=${GPUS[$i]}
      local log="$ROOT/exp/logs/target-worker-gpu${gpu}-shard${i}-${phase}.log"
      if [ ! -f "$log" ] || ! grep -q "target worker GPU=${gpu} done" "$log"; then
        pending=1
      fi
    done
    [ "$pending" -eq 0 ] && return 0
    sleep 30
  done
}

if [ "$MODE" = all ] && [ "${TARGET_MASTER:-0}" != 1 ]; then
  if tmux has-session -t qam-target-master 2>/dev/null; then
    echo "tmux session qam-target-master already exists" >&2
    exit 1
  fi
  gpu_args="${GPUS[*]}"
  tmux new-session -d -s qam-target-master \
    "export TARGET_MASTER=1; cd $ROOT && bash scripts/launch_target_screen_tmux.sh all $gpu_args; echo EXIT:\$?; exec bash"
  echo "started qam-target-master (tmux attach -t qam-target-master)"
  exit 0
fi

if [ "$MODE" = all ]; then
  start_workers smoke
  wait_workers smoke
  if grep -q "done fail=1" "$ROOT"/exp/logs/target-worker-gpu*-smoke.log 2>/dev/null; then
    echo "smoke failed; full screen will not start" >&2
    exit 1
  fi
  start_workers full
elif [ "$MODE" = smoke ] || [ "$MODE" = full ]; then
  start_workers "$MODE"
else
  echo "mode must be all, smoke, or full" >&2
  exit 2
fi
