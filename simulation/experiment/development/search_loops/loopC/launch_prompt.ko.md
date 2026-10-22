> Redacted copy of the authors' working document: lines about team messaging and notifications removed; names, hosts and paths replaced.

너는 무인 "belief update 루프 4차"(3차 L6과 autoresearch 플러그인 최종의 좋은 부분을 합친 출발점 + autoresearch)의 한 번 호출이다. 사람은 없다. 질문하지 말고, 지시서에 없는 판단이 필요하면 BLOCKED로 적고 끝낸다.

[removed: team coordination]
2. PROGRESS의 STEP에서 **다음 일 하나**를 한다(대략 60분 이내). 기다리는 중이면 상태만 확인한다. 끝난 것은 로그·산출물을 확인하고 다음 단계로 넘긴다.
3. PROGRESS.md를 갱신한다: STATUS, STEP, `NEXT: now|wait`, UPDATED(`date -u '+%F %H:%M'`), 로그 한 줄(무엇을 했고 수치). 머리 줄 `MODEL: opus`(항상), `WAIT_TMUX: <세션>`(tmux 작업을 기다릴 때만, 아니면 빈칸)을 반드시 적는다.
4. 바꿨으면 테스트(`cd simulation/belief_v2 && python3 -m pytest -q tests/`) 통과 후 `<branch>`에 커밋한다. **push 하지 않는다.** `git add`는 파일 지정만(`-A`·`.` 금지). 커밋 메시지 끝: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. PROGRESS만 바뀌어도 커밋.
5. 종료.

규칙(자세한 것은 지시서 §6–§7):
- 평가는 tmux `bv4_eval`에서 `bash loop/eval4.sh <impl> <run>`(빈 GPU를 5분마다 기다림, GEN_RUNS=5), 로그는 파일로, `WAIT_TMUX: bv4_eval`로 기다린다. 다른 사람이 쓰는 GPU는 건드리지 않는다. <server>는 쓰지 않는다.
- **시험 세트(`~/belief_v2_sealed/`)는 열지 않고 평가하지 않는다.**
[removed: team coordination]
- 같은 단계 3회 연속 실패, 권한 거부, 측정·세트를 바꿔야 하는 판단 → `STATUS: BLOCKED`와 사유, ATTENTION 항목.
- F 끝: 지시서 §5대로 보고서·그림·최종 파일·메모리, `pool_down`, `STATUS: DONE`.
