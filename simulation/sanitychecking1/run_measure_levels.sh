#!/usr/bin/env bash
# Measure all 10 Avery persona levels' ISMI-29 (qwen3:32b, 30 reps each,
# judge ctx 4096) across N ollama endpoints, one job queue shared by all
# workers so nothing gets measured twice.
#
# Usage: ./run_measure_levels.sh host:port [host:port ...]
set -u
cd "$(dirname "$0")"
OUT_DIR=outputs/level_ismi29_30reps
INPUT=outputs/avery_10level_personas.json
REPS=30
NUM_CTX=4096

ENDPOINTS=("$@")
if [ "${#ENDPOINTS[@]}" -eq 0 ]; then
  echo "usage: $0 host:port [host:port ...]" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
QUEUE_DIR=$(mktemp -d)
JOBLIST="$QUEUE_DIR/jobs.txt"
: > "$JOBLIST"
for i in 1 2 3 4 5 6 7 8 9 10; do
  echo "level_${i}" >> "$JOBLIST"
done

LOCK="$QUEUE_DIR/lock"
next_job() {
  while ! mkdir "$LOCK" 2>/dev/null; do sleep 0.2; done
  line=$(head -n 1 "$JOBLIST")
  if [ -n "$line" ]; then
    tail -n +2 "$JOBLIST" > "$JOBLIST.tmp" && mv "$JOBLIST.tmp" "$JOBLIST"
  fi
  rmdir "$LOCK"
  echo "$line"
}

worker() {
  ep="$1"; wid="$2"
  while true; do
    level=$(next_job)
    [ -z "$level" ] && break
    log="$OUT_DIR/${level}.log"
    echo "[worker $wid @ $ep] START $level"
    python3 measure_single_level.py \
      --input "$INPUT" --level "$level" \
      --model qwen3:32b --base-url "http://${ep}/v1" \
      --num-ctx "$NUM_CTX" --reps "$REPS" \
      --out "$OUT_DIR/${level}.json" > "$log" 2>&1
    if [ $? -eq 0 ]; then echo "[worker $wid @ $ep] DONE $level"
    else echo "[worker $wid @ $ep] FAILED $level"; fi
  done
  echo "[worker $wid @ $ep] queue empty, exiting"
}

for i in "${!ENDPOINTS[@]}"; do
  worker "${ENDPOINTS[$i]}" "$((i + 1))" &
done
wait
echo "[queue] all jobs done"
rm -rf "$QUEUE_DIR"
