> Redacted copy of the authors' working document: lines about team messaging and notifications removed; names, hosts and paths replaced.

# belief update 루프 3차 — 데이터셋 v3 확정 + autoresearch 루프 지시서 (확정, 2026-10-13)

> 사용자 결정(10-13): 교수님 권고로 autoresearch 방식(고칠 파일 하나·지표 하나·유지/되돌리기). 데이터셋 v3가 나오고
[removed: team coordination]
> 루프 출발점은 **논문 게이트+재작성**(paper_rq1/belief_update.py), 구조까지 자유 수정, **48시간 또는 20회차**.
> 데이터셋 계획: `2026-10-13-belief-eval-dataset.md`.
> **사용량 (10-13 22:50 사용자 요청)**: Fable은 v4 진단·수정에만. 이상적 노트·검토자·교차검증 서브에이전트는 Sonnet, 대화 생성만 Opus. 이전 루프: `2026-10-11-belief-update-loop.md`, `2026-10-12-belief-update-loop2.md`.

## 1. 단계
| 단계 | 할 일 | 모델 |
|---|---|---|
| F0 | 데이터셋 파이프라인(tmux `bv3_data`)이 끝나길 기다린다 (`WAIT_TMUX: bv3_data`) | — |
[removed: team coordination]
| F2 | 기준선(§3) → ATTENTION(저자만) | opus |
| L1–L20 | 개선 회차(§4) | **v4 진단·수정만 fable**, 나머지 opus |
| F | 최종 후보 고정·보고서·메모리·`STATUS: DONE` | opus |

## 2. F1 — 데이터셋 확정
1. `results/v3_check/SUMMARY_dev.md`, `SUMMARY_test.md`, `results/v3_s0/SELECT.md`, `_pipeline.log`를 읽는다.
2. **교차검증(개발 16개)**: 채택된 시도(`check_v3.py summary --split dev`의 accepted, 같은 대화는 최신 시도)를 대상으로
   **Agent 도구로 서브에이전트 2개를 독립적으로**(model: sonnet — 사용량 절약, 10-13) 띄운다. 각자 `testset_v3/candidates_dev.jsonl`에서 그 시도들을 읽고, 칸의 의도
   (README 표)와 `check_v3.REVIEW_CODES`로 대화마다 pass / 문제 코드 / 한 줄 근거를 낸다. 둘 중 하나라도 major를 낸 대화는
   메인이 직접 읽고 판정한다. 메인은 그 밖에도 유형마다 하나 이상을 직접 읽는다.
   - 기각한 대화는 tmux `bv3_regen`에서 `python3 gen_v3.py --split dev --only <ids>` 뒤 `check_v3.py` direction·oracle·judge·review
     (judge는 `source loop/<server>_ollama.sh && <server>_up` 후 `--base-url http://localhost:11888/v1`)로 다시 만들고 검사한다
     (대화당 시도 합계 5회까지). 그래도 안 되면 그 대화는 빼고 ATTENTION에 적는다.
3. **시험 8개는 읽지 않는다**(§6). `SUMMARY_test.md`의 수치·코드만 본다. 미채택이 있으면 같은 방식으로 재생성·재검사(합계 5회).
   그래도 빠지면 빠진 채로 진행하고 ATTENTION.
4. `python3 check_v3.py summary --split dev --assemble`, 같은 것 `--split test`, `python3 manifest_v3.py`.
5. 테스트 통과 후 커밋·push: `testset_v3/`(dev.jsonl, pre/, s0_pooled.json, MANIFEST.json, README.md, candidates_dev.jsonl,
   registers.json), `results/v3_check/`(개발 검사 JSON·SUMMARY·교차검증 기록 `CROSSCHECK.md`), `results/v3_s0/`(개발).
   봉인 파일(`~/belief_v2_sealed/`)은 절대 커밋하지 않는다.
[removed: team coordination]

## 3. F2 — 기준선과 잡음
- tmux `bv3_eval`에서 순서대로: `bash loop/eval_v3.sh paper base_paper`, `bash loop/eval_v3.sh paper base_paper_r2`,
  `bash loop/eval_v3.sh v3 base_v3`. (`WAIT_TMUX: bv3_eval`)
- **잡음 폭 = |E(base_paper) − E(base_paper_r2)|**. 이보다 작은 개선은 "나아졌다"고 하지 않는다. 최소 0.02로 본다. (10-14: 유지 판정은 §4의 두 번 평균·0.04 규칙으로 바뀜)
- `results/v3_runs/BASELINE.md`에 세 결과(E, 칸별 Δ, 지킴 조건)와 잡음 폭. 저자에게 ATTENTION(다른 팀원에게는 보내지 않는다).

## 4. L — 개선 회차 (autoresearch)
- **고칠 파일은 `simulation/belief_v2/belief_update_v4.py` 하나**(+ 그 테스트 `tests/test_v4_*.py`). 출발은 논문 코드 사본.
  게이트·재작성·호출 수·노트 형식 등 구조까지 자유롭게 바꿔도 된다. 필요하면 선행연구를 찾아본다(WebSearch·WebFetch).
- 한 회차: ① 직전 결과에서 가장 나쁜 칸의 노트와 대화를 읽고 원인 가설 하나 ② PROGRESS에 가설·바꿀 것·예상 사전 등록
  ③ v4 수정 → 테스트 → 커밋·push ④ tmux `bv3_eval`에서 `bash loop/eval_v3.sh v4 l3_iter_NN` ⑤ 채점 →
  **유지 조건 (10-14 14:30 KST 사용자 결정으로 변경)**: 수정본은 항상 두 번 돈다 — `l3_iter_NN`과 같은 코드의 `l3_iter_NN_confirm`
  (첫 실행에서 지킴 조건 실패 또는 E가 현재 최고 평균보다 높으면 confirm 없이 되돌려도 된다). 판정은 **두 번 평균 E**로 한다:
  두 실행 모두 지킴 조건 통과 **그리고** 두 번 평균 E ≤ (현재 최고의 두 번 평균 E − 0.04)이면 유지, 아니면 `git revert`.
  0.04 = 두 번 평균의 흔들림(L2 두 실행 차이 0.06 / √2). 기준선 잡음 폭 0.021은 논문 코드처럼 노트를 한 방향으로만 바꾸는
  방법의 흔들림이라 새 버전들(실행 간 0.06)에는 작다. 현재 최고 = L2(84651f87), 두 번 평균 0.160 → 다음 유지 문턱 0.120.
  이미 진행 중인 회차(L3)도 이 규칙으로 판정한다.
- 각 수정이 악화·감소·유지에 **대칭으로** 작용하는지 PROGRESS에 한 줄로 적는다.
- 종료: 48시간(L1 시작부터) 또는 20회차, 또는 연속 5회차 개선 없음, 또는 **조기 종료(사용자 결정 10-14)**: 유지된 v4가
  지킴 조건 통과 + E ≤ 0.15 + 그 confirm 재실행에서도 지킴 조건 통과·E ≤ 0.15(두 번 평균도 ≤ 0.15)이면 그 회차로 루프를 끝내고 F로 간다.
  기준 0.15는 개발 16개의 이상적 노트(데이터셋 검사 때 Claude가 갱신한 노트) E 0.123에 근접한 수준이다(MANIFEST oracle_delta_mean으로 계산).
  조기 종료하면 ATTENTION에 알린다.

[removed: team coordination]
  저자에게는 F1 요약과 ATTENTION(§7)으로만.

## 6. 금지
- **시험 세트**: `~/belief_v2_sealed/` 안의 파일을 Read·cat·head 등으로 열지 않는다. 그 경로를 다루는 것은 스크립트
  (`gen_v3.py`, `check_v3.py`, `manifest_v3.py`, `s0_select.py`)뿐이고, 그 출력(숫자·코드)만 본다. 시험 세트로 평가하지 않는다
  (`eval_v3.sh ... test` 금지 — 최종 비교는 사용자가 정한다).
- v4 프롬프트에 개발 대화의 표현·사람·사건을 넣지 않는다. 유형·목표·방향을 알려주는 지시 금지. 방향을 한쪽으로 미는 지시 금지
  (대칭적인 규칙만). 응답자 신원 사용 금지. 상대 경험의 전유 금지.
- 측정·세트·채점 불변: `harness.py`, `score_v3.py`, `judge_notes.py`, `check_v3.py`, `gen_v3.py`, `v3data.py`, `testset_v3/`(F1 확정 후),
  `s0_*.py`, 이 지시서, `main_experiment/`, `RQ3_crowd/`, `sanitychecking2/`, 다른 세션의 미커밋 파일
  (`RQ3_crowd/autonomy_grid.py`, `RQ3_crowd/tests/test_autonomy_grid.py`, `docs/superpowers/plans/2026-09-29-rq3-autonomy-probe.md`).
- GPU: <server>의 빈 카드 한 장(`loop/<server>_ollama.sh`가 GPU0→1→2→3 순으로 빈 카드를 골라 띄운다, 10-14). 모두 바쁘면 BLOCKED 말고 30분 뒤 다시(`NEXT: wait`).
  루프가 끝나면 `<server>_down`.

[removed: team coordination]
- F1 확정 결과, F2 기준선, 대화가 빠졌을 때, 처음으로 유지된 개선, 결과가 이상할 때(잡음 넘는 퇴보·유지 칸 표류·전유 재발),
  큰 구조 변경(호출 구조·게이트 제거 등)을 할 때, F 진입 전 최종 후보.

## 8. 이어가기 L7–L10 (사용자 결정 2026-10-15)

> 10-15 autoresearch 플러그인 시범(1라운드: 같은 출발·같은 평가, 사전 지식 없음, 10회차)과 **회차 수를 맞춰** 비교하려고,
> 조기 종료로 6회차에서 끝난 이 루프를 L6 위에서 **10회차까지** 이어 간다. 이 절이 §1·§4·§6·§7과 다르면 이 절을 따른다.

- **출발**: 현재 최고 = L6 (4a6e232b, `belief_update_v4.py`가 그대로), 두 번 평균 E 0.1145. 연속 무개선 0에서 시작.
- **회차**: L7, L8, L9, L10 네 번. L10이 끝나면 F′로 간다. 48시간 기준과 §4의 조기 종료(E ≤ 0.15)는 **쓰지 않는다**.
  연속 5회차 무개선 규칙은 남은 회차가 4라 걸리지 않는다.
- **유지 조건은 §4 그대로**(두 실행 모두 지킴 조건 통과 그리고 두 번 평균 E ≤ 현재 최고 두 번 평균 − 0.04 → 지금 문턱 **0.0745**).
  첫 실행에서 지킴 실패 또는 E > 0.1145이면 confirm 없이 되돌린다. 문턱이 높으니, 유지가 안 돼도 각 회차의 진단과
  칸별 수치를 정확히 남기는 것이 이 이어가기의 주 산출물이다.
- **모델: 전부 opus** (사용량 — 주간 한도가 얼마 남지 않음, 10-15 사용자 결정). PROGRESS의 `MODEL:`은 항상 `opus`로 적는다.
  서브에이전트를 쓰면 sonnet. 평가 한 번에 호출 여러 번을 쓰지 않도록, 기다릴 때는 `WAIT_TMUX`로 셸이 기다리게 한다.
- **GPU 풀 (10-15 07:15 사용자 결정: <server> 비우고 <server>만 — 카드 2장 = 포트 11889·11890, 각 32b+14b 전량 GPU)**: 평가는 반드시 POOL을 명령에 넣어 띄운다(tmux는 호출 쪽 환경을 넘기지 않는다):
  `tmux new -d -s bv3_eval "cd <repo>/simulation/belief_v2 && POOL='<server>:2 <server>:1' bash loop/eval_v3.sh v4 l3_iter_NN > results/v3_runs/_l3_iter_NN.log 2>&1"`
  (confirm은 `l3_iter_NN_confirm`). 평가 한 번 약 25–30분. F′ 끝에 `cd simulation/belief_v2 && source loop/gpu_pool.sh && pool_down`.
- **읽지 않는 것**: 플러그인 시범의 파일(`simulation/belief_v2/autoresearch/`, `belief_update_v5.py`, `belief_update_v6.py`,
  `results/v3_runs/ar_*`, `ar2_*`). 두 방법을 독립적으로 비교하려는 것이다.
[removed: team coordination]
- **F′**: `results/loop3/REPORT.md` 끝에 `## 이어가기 L7–L10 (10-15)` 절을 덧붙인다(앞 내용은 고치지 않는다): 결론 먼저,
  회차 표(가설·E 1차/confirm·판정), 칸별 Δ, copy_scan 1인칭 복사·2인칭, 남은 문제. 유지가 있었으면 `final/belief_update_v4_<hash>.py`를
  새로 두고, 없으면 최종 후보는 L6 그대로라고 적는다. 메모리 `project_1013_eval_dataset_v3.md`에 한 단락 덧붙인다. `pool_down`, `STATUS: DONE`.
