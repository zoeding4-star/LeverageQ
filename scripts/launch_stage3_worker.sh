#!/usr/bin/env bash
# Sequential Stage-3 jobs on ONE allowed idle GPU. Refuses occupied cards.
# Usage: bash launch_stage3_worker.sh <gpu> <shard_index> <n_shards> [smoke|full]
set -euo pipefail

GPU="${1:?idle gpu index}"
SHARD="${2:?shard index 0..n-1}"
NSHARD="${3:-3}"
MODE="${4:-full}"
ROOT=/mnt/zoe/projects/qam
PY=/mnt/zoe/conda-envs/qam/bin/python
LOGDIR="$ROOT/exp/logs"
mkdir -p "$LOGDIR"
MASTER_LOG="$LOGDIR/stage3-worker-gpu${GPU}-shard${SHARD}-${MODE}.log"

case "$GPU" in
  1|4|5) ;;
  *)
    echo "GPU $GPU is not in the allowed idle set 1,4,5. Refusing." >&2
    exit 2
    ;;
esac

cd "$ROOT"
echo "[$(date -Is)] worker GPU=$GPU shard=$SHARD/$NSHARD mode=$MODE" | tee -a "$MASTER_LOG"

JOB_ARGS=(--shard "$SHARD" --n-shards "$NSHARD")
if [ "$MODE" = "smoke" ]; then
  JOB_ARGS+=(--debug-only)
fi
mapfile -t JOBS < <("$PY" "$ROOT/stage3/jobs.py" "${JOB_ARGS[@]}")
echo "[$(date -Is)] ${#JOBS[@]} jobs: ${JOBS[*]}" | tee -a "$MASTER_LOG"

fail=0
for job in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  set -- $job
  MASK="$1"
  SEED="$2"
  BUDGET="$3"
  echo "[$(date -Is)] START mask=$MASK seed=$SEED budget=$BUDGET on GPU $GPU" | tee -a "$MASTER_LOG"
  bash "$ROOT/scripts/wait_idle_gpu.sh" "$GPU" 30 | tee -a "$MASTER_LOG"
  if bash "$ROOT/scripts/launch_stage3.sh" "$GPU" "$MASK" "$SEED" "$MODE" "$BUDGET"; then
    echo "[$(date -Is)] OK mask=$MASK seed=$SEED" | tee -a "$MASTER_LOG"
  else
    echo "[$(date -Is)] FAIL mask=$MASK seed=$SEED (exit $?). Continuing remaining jobs." | tee -a "$MASTER_LOG"
    fail=1
  fi
done

echo "[$(date -Is)] worker GPU=$GPU done fail=$fail" | tee -a "$MASTER_LOG"
exit "$fail"
