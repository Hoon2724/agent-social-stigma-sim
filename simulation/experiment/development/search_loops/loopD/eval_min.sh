#!/bin/bash
# Redacted copy: lines about team messaging and notifications removed; names, hosts and paths replaced.
# Autoresearch (minimal start) eval of simulation/belief_v2/belief_update_v8.py on dataset v3 dev (16 dialogues),
# 5 generations per dialogue, judge 3x. Runs in tmux (an eval takes 30-90 min depending on free GPUs):
#   bash autoresearch/eval_min.sh start <run>   |   bash autoresearch/eval_min.sh wait <run>
# Run names: arm_iter_NN, arm_iter_NN_confirm. GPU servers: whatever tmux gpu_grab holds (<server> cards,
# <server> pair on 11908); <server> is not used.
set -u
cd "$(dirname "$0")/.."
RUN=${2:?run name}
case "$RUN" in arm_*) ;; *) echo "run names start with arm_"; exit 2;; esac
case "$1" in
  start)
    [ -e "results/v3_runs/$RUN/_score.json" ] && { echo "run name $RUN already used -- pick a new one"; exit 2; }
    P="<server>:4"   # 10-16 사용자: <server>만
    tmux new -d -s "ar_$RUN" "POOL='$P' GEN_RUNS=5 bash loop/eval_v3.sh belief_update_v8.py $RUN > results/v3_runs/_$RUN.log 2>&1"
    echo "STARTED $RUN (tmux ar_$RUN)";;
  wait)
    for _ in $(seq 1 54); do tmux has-session -t "ar_$RUN" 2>/dev/null || break; sleep 10; done
    if tmux has-session -t "ar_$RUN" 2>/dev/null; then
      echo "RUNNING $RUN ($(ls results/v3_runs/$RUN/*.json 2>/dev/null | grep -vc _score)/16 dialogues done) -- call wait again"
    else
      python3 autoresearch/score_2.py "$RUN"
    fi;;
  *) echo "usage: eval_min.sh start|wait <run_name>"; exit 2;;
esac
