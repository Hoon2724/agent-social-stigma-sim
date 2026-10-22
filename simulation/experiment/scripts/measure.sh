#!/bin/bash
# Score every run record under a directory with the ISMI-29 judge (qwen3:32b, 3 calls per belief state, T 0.3),
# from the belief state alone, as in the paper; then export the long table used by analysis/.
#   OLLAMA_URL=http://localhost:11434/v1 bash scripts/measure.sh results/rerun [level ...]
# Only finished records are scored (a record is rewritten after every session while its run is going).
set -euo pipefail
ROOT=$(cd "${1:?run directory, e.g. results/rerun}" && pwd); shift
LEVELS="${*:-}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
URL=${OLLAMA_URL:-http://localhost:11434/v1}
PY=${PYTHON:-python3}
cd "$HERE/agents"
if [ -n "$LEVELS" ]; then
  RUNS=$(for L in $LEVELS; do find "$ROOT" -path "*/level_${L}/*" -name 'run_[0-9][0-9].json'; done | sort)
else
  RUNS=$(find "$ROOT" -name 'run_[0-9][0-9].json' | sort)
fi
for run in $RUNS; do
  out="${run%.json}.beliefonly.qwen.ismi29.json"
  [ -e "$out" ] && continue
  "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("status") != "complete")' "$run" \
    || { echo "skip unfinished $run"; continue; }
  "$PY" measure_ismi29_run.py --run "$run" --out "$out" --personas outputs_v2/personas.json \
    --model qwen3:32b --reps 3 --temperature 0.3 --pre-from-session0 --belief-update-only --base-url "$URL"
  "$PY" check_ismi29.py "$out" || mv -f "$out" "$out.bad"    # a session whose three calls all failed: rerun
done
"$PY" export_beliefonly_ismi_long.py --root "$ROOT" --out "$ROOT/ismi_long.csv"
echo "wrote $ROOT/ismi_long.csv"
