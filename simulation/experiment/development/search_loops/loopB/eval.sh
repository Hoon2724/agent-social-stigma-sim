#!/bin/bash
# Redacted copy: lines about team messaging and notifications removed; names, hosts and paths replaced.
# Autoresearch eval for simulation/belief_v2/belief_update_v5.py on dataset v3 dev (16 dialogues).
# One evaluation takes ~45-60 min on the lab GPU pool, longer than one tool call, so it runs in tmux:
#   bash autoresearch/eval.sh start <run_name>   # starts it, returns at once
#   bash autoresearch/eval.sh wait  <run_name>   # waits up to 9 min; prints RUNNING or the SCORE block
# Run names: ar_iter_NN, ar_iter_NN_confirm. Results land in results/v3_runs/<run_name>/.
set -u
cd "$(dirname "$0")/.."
export POOL="${POOL:-<server>:2 <server>:2}"   # 10-15 user: use <server> (and <server>)
RUN=${2:?run name}
case "$1" in
  start)
    [ -e "results/v3_runs/$RUN/_score.json" ] && { echo "run name $RUN already used -- pick a new one"; exit 2; }
    tmux new -d -s "ar_$RUN" "POOL='$POOL' bash loop/eval_v3.sh belief_update_v5.py $RUN > results/v3_runs/_$RUN.log 2>&1"   # pass POOL in: a running tmux server does not inherit the caller env
    echo "STARTED $RUN (tmux ar_$RUN)";;
  wait)
    for _ in $(seq 1 54); do tmux has-session -t "ar_$RUN" 2>/dev/null || break; sleep 10; done
    if tmux has-session -t "ar_$RUN" 2>/dev/null; then
      echo "RUNNING $RUN ($(ls results/v3_runs/$RUN/*.json 2>/dev/null | grep -vc _score)/16 dialogues done) -- call wait again"
    else
      python3 autoresearch/score.py "$RUN"
    fi;;
  *) echo "usage: eval.sh start|wait <run_name>"; exit 2;;
esac
