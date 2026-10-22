#!/bin/bash
# Run one persona x severity cell of the 1:1 experiment, both conditions, three runs each (as in the paper).
#   OLLAMA_URL=http://localhost:11434/v1 bash scripts/run_cell.sh 270            # Noor, severe
# Levels: 110 111 112 (Avery severe/moderate/mild), 240-242 (Jordan), 250-252 (Riley), 260-262 (Ravi), 270-272 (Noor).
# Needs: an Ollama server with qwen3:14b, and agents/outputs_v2/personas.json filled with the style examples
# (agents/STYLE_EXAMPLES.md). Output: results/rerun/{peer,general}/level_<L>/run_0{1,2,3}.json
set -euo pipefail
LEVELS="${*:?give one or more levels, e.g. 270}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$HERE/results/rerun}
URL=${OLLAMA_URL:-http://localhost:11434/v1}
PY=${PYTHON:-python3}
SESSIONS=${SESSIONS:-8}; ROUNDS=${ROUNDS:-5}; RUNS=${RUNS:-3}
cd "$HERE/agents"
"$PY" fill_style_examples.py --check
"$PY" run_dialogues.py --personas outputs_v2/personas.json --levels $LEVELS \
  --agent-type peer --n-general 0 --dynamic-peer-backstory \
  --sessions "$SESSIONS" --rounds "$ROUNDS" --runs "$RUNS" --track-belief --model qwen3:14b \
  --pre-narratives config/session0_beliefs.json \
  --output-dir "$OUT/peer" --num-ctx 8192 --num-gpu 99 --base-url "$URL"
"$PY" run_dialogues.py --personas outputs_v2/personas.json --levels $LEVELS \
  --agent-type general --n-general 1 --general-variant restrained \
  --sessions "$SESSIONS" --rounds "$ROUNDS" --runs "$RUNS" --track-belief --model qwen3:14b \
  --pre-narratives config/session0_beliefs.json \
  --output-dir "$OUT/general" --num-ctx 8192 --num-gpu 99 --base-url "$URL"
