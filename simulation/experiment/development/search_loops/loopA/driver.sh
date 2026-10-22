#!/bin/bash
# Redacted copy: lines about team messaging and notifications removed; names, hosts and paths replaced.
# belief update 개선 루프 드라이버 — <server> tmux(bv3_loop) 안에서 돈다 — 3차(데이터셋 v3 확정+autoresearch). `claude -p`(Fable)를 한 단계씩 반복 호출한다.
#   tmux new -d -s bv3_loop 'bash simulation/belief_v2/loop/driver3.sh'
# 멈춤: PROGRESS 첫 줄이 DONE/BLOCKED, 72시간 경과, 연속 3회 호출 실패,
#       또는 loop/STOP3 파일이 생기면(사람이 끄고 싶을 때 touch — 다음 회차에 멈춘다).
# 사용량 절약(사용자 요청 10-11): 기다리는 동안은 claude를 부르지 않는다 — PROGRESS의
#   WAIT_JOBS: <slurm id,...>  가 아직 큐에 있으면, WAIT_TMUX: <세션>  이 아직 살아 있으면 셸이 그냥 다시 잔다.
# 모델은 PROGRESS의 `MODEL: fable|opus`를 따른다(기본 opus). Fable은 진단·코드 수정·보고서 단계에만.
set -u
REPO=<repo>
HERE=$REPO/simulation/belief_v2/loop
PROGRESS=$REPO/simulation/belief_v2/results/loop3/PROGRESS.md
# [removed: team coordination]
START=$(date +%s); LIMIT=$((72*3600)); FAILS=0; I=0
cd "$REPO"

while true; do
  I=$((I+1)); NOW=$(date '+%F %H:%M:%S')
  status=$(head -1 "$PROGRESS" | awk '{print $2}')
  case "$status" in DONE|BLOCKED) echo "[$NOW] $status -- driver exits"; break;; esac
  if [ -e "$HERE/STOP3" ]; then echo "[$NOW] STOP file -- driver exits"; break; fi
  if [ $(( $(date +%s) - START )) -gt $LIMIT ]; then
    echo "[$NOW] 72h limit"; sed -i '1s/.*/STATUS: BLOCKED/' "$PROGRESS"
    printf '\n- %s 드라이버: 72시간 한도 도달\n' "$NOW" >> "$PROGRESS"; break
  fi

  wait_jobs=$(grep -m1 '^WAIT_JOBS:' "$PROGRESS" | cut -d: -f2- | tr -d ' ')
  if [ -n "$wait_jobs" ] && timeout 60 ssh <cluster-login> "squeue -h -j $wait_jobs 2>/dev/null" | grep -q .; then
    I=$((I-1)); sleep 600; continue
  fi
  wait_tmux=$(grep -m1 '^WAIT_TMUX:' "$PROGRESS" | cut -d: -f2- | tr -d ' ')
  if [ -n "$wait_tmux" ] && tmux has-session -t "$wait_tmux" 2>/dev/null; then
    I=$((I-1)); sleep 300; continue
  fi
  case "$(grep -m1 '^MODEL:' "$PROGRESS" | awk '{print $2}')" in
    fable) MODEL=claude-fable-5-1 ;;
    *)     MODEL=claude-opus-5-5 ;;
  esac

  echo "[$NOW] iteration $I model=$MODEL (step: $(sed -n 2p "$PROGRESS"))"
  if claude -p "$(cat "$HERE/prompt3.md")" \
      --model "$MODEL" \
      --permission-mode acceptEdits \
      --settings "$HERE/settings3.json" \
      --add-dir "$MEM" \
      > "$HERE/logs/l3_call_$(printf %03d $I).log" 2>&1; then
    FAILS=0
  elif grep -q "session limit\|usage limit\|rate limit" "$HERE/logs/l3_call_$(printf %03d $I).log"; then
    echo "  usage limit -- sleep 30m (not counted as a failure)"; I=$((I-1)); sleep 1800; continue
  else
    FAILS=$((FAILS+1)); echo "  call failed ($FAILS)"
    if [ $FAILS -ge 3 ]; then
      sed -i '1s/.*/STATUS: BLOCKED/' "$PROGRESS"
      printf '\n- %s 드라이버: claude 호출 3회 연속 실패, loop/logs/ 확인\n' "$NOW" >> "$PROGRESS"; break
    fi
  fi
  tail -3 "$HERE/logs/l3_call_$(printf %03d $I).log" | sed 's/^/  | /'

  if grep -q "^NEXT: now" "$PROGRESS"; then sleep 60; else sleep 900; fi
done
