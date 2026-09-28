#!/usr/bin/env bash
# Sequential target-screen jobs on one allowed GPU. Never kills a process.
set -euo pipefail

GPU="${1:?gpu}"
SHARD="${2:?shard}"
NSHARD="${3:?number of shards}"
MODE="${4:-full}"
ROOT=/mnt/zoe/projects/qam
PY=/mnt/zoe/conda-envs/qam/bin/python
LOG="$ROOT/exp/logs/target-worker-gpu${GPU}-shard${SHARD}-${MODE}.log"

case "$GPU" in
  1|4|5) ;;
  *) echo "GPU $GPU is outside the allowed set 1,4,5" >&2; exit 2 ;;
esac

cd "$ROOT"
JOB_OUTPUT=$("$PY" stage3_target/jobs.py --shard "$SHARD" --n-shards "$NSHARD")
mapfile -t JOBS <<< "$JOB_OUTPUT"
if [ "${#JOBS[@]}" -eq 0 ] || [ -z "${JOBS[0]}" ]; then
  echo "job list is empty; refusing to report success" >&2
  exit 2
fi
echo "[$(date -Is)] target worker GPU=$GPU shard=$SHARD/$NSHARD mode=$MODE jobs=${JOBS[*]}" | tee -a "$LOG"
fail=0
for mask in "${JOBS[@]}"; do
  echo "[$(date -Is)] START $mask on GPU $GPU" | tee -a "$LOG"
  bash "$ROOT/scripts/wait_idle_gpu.sh" "$GPU" 30 | tee -a "$LOG"
  if bash "$ROOT/scripts/launch_target_screen.sh" "$GPU" "$mask" "$MODE"; then
    echo "[$(date -Is)] OK $mask" | tee -a "$LOG"
  else
    status=$?
    echo "[$(date -Is)] FAIL $mask exit=$status" | tee -a "$LOG"
    fail=1
  fi
done
echo "[$(date -Is)] target worker GPU=$GPU done fail=$fail" | tee -a "$LOG"
exit "$fail"
