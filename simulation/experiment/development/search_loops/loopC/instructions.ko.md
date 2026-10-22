> Redacted copy of the authors' working document: lines about team messaging and notifications removed; names, hosts and paths replaced.

# belief update 루프 4차 — 두 루프의 좋은 부분을 합친 출발점 + autoresearch (확정, 2026-10-15)

> 사용자 결정(10-15): 3차 루프(L6, 4a6e232b)와 autoresearch 플러그인 1라운드(edf31ae3)가 둘 다 MAE 0.11–0.14에서
> 멈췄고 차이는 흔들림 안이다. 두 설계의 좋은 부분을 합친 출발점을 만들고, 그 위에서 다시 루프를 돈다. tmux 무인 실행.
> 같은 데이터셋 v3 개발 16개, 같은 판정기. 시험 세트(`~/belief_v2_sealed/`)는 열지도 평가하지도 않는다.
> 배경: `results/loop3/REPORT.md`(끝 절 "이어가기 L7–L10" 포함), `autoresearch/autoresearch_dashboard.md`(플러그인 최종 보고),
> `results/loop3/compare_E_curve.png`, 메모리 `project_1013_eval_dataset_v3.md`.

## 1. 단계
| 단계 | 할 일 | 모델 |
|---|---|---|
| C0 | 합친 출발점 `belief_update_v7.py` 작성(§2) → 테스트 → 커밋 | opus |
| C1 | 출발점 평가 2번(§3) → 판단 → ATTENTION | opus |
| L1–L10 | 개선 회차(§4) | opus |
| F | 최종 후보 고정·보고서·그림·메모리·`STATUS: DONE` | opus |

모델은 **전부 opus**(10-15 사용자: 주간 한도 86% 사용 — Fable 쓰지 않음). 서브에이전트를 쓰면 sonnet.

## 2. C0 — 합친 출발점 (`simulation/belief_v2/belief_update_v7.py`)
**뼈대는 플러그인 최종**(`final/belief_update_v5_edf31ae.py`)에서 가져온다. 이유: 닿은 문장만 고쳐 써서 건드리지 않은 문장이
사라지지 않는다(L6은 26–35% 유실), 근거마다 방향(확인/반박/무관)이 기록돼 추적된다, 출처 확인이 코드라 확정적이다.
여기에 **L6(`final/belief_update_v4_4a6e232.py`)에서 세 가지**를 옮긴다:
1. **나에 대한 단정 통로**(L4+L6의 `labels_on_me` / `_LABEL_SYSTEM` / 꼬리표 루프): 노트에 자리가 없는, 내가 한 구체 행동에 박아
   남이 나를 "~다/~가 아니다"라고 한 말. 조건은 L6 그대로 — **번호 통로가 아무것도 못 연 주에만, 한 주에 하나**.
   문장 단위 구조에서는 **추가가 아니라 기존 문장 하나를 대체**한다: 그 단정이 다루는 두려움과 가장 가까운 문장(없으면
   노트에서 자기 평가를 말하는 문장 중 하나)을 골라, 단정의 방향(참이다 / 모른다 / 참이 아니다 세 착지)으로 다시 쓴다.
   노트 문장 수는 늘리지 않는다. 어느 문장을 골랐는지 trace에 남긴다. 대칭: 어느 부호의 단정이든 같은 조건.
2. **상대의 자기 이야기 배제**(L6 `_own_story`): 근거가 1인칭(I / I've / my / when I …)으로 시작하고 2인칭이 없으면 버린다 —
   **번호 통로와 단정 통로 모두에**(L6은 단정 통로에만 걸어서 번호 통로 전유가 남았다). 플러그인 `_direction` 프롬프트의
   "A report of the speaker's own experience counts…" 예시 두 개는 지운다(남의 경험을 근거로 인정하는 지시).
3. **프롬프트 예시 문구 재생산 차단**(L1 `_PROMPT_EXAMPLE_PHRASES`): 프롬프트에 든 예시 문구를 재생산한 근거는 버린다.

그리고 **개발 대화 문구 유출 검사**를 테스트로 고정한다(플러그인 8회차 교훈: 예시 문구가 개발 대화 댓글과 6단어 연속 일치해
점수가 부풀었다): `tests/test_v7_leak.py` — v7의 모든 프롬프트 문자열에서 `testset_v3/dev.jsonl`의 어떤 발화와도
**6단어 이상 연속 일치**가 없어야 통과. 이후 모든 회차에서 이 테스트가 통과해야 한다.

공개 함수(`initial_selfreport`, `belief_update_avg`, `narrative_of`, `update`)의 서명·반환 키, 하니스가 저장하는 세션 필드
(`changed`, `grounded_in`, `belief_update`)는 그대로. 테스트 `tests/test_v7_*.py`(대칭 짝 수, 금지어, 유출, 단정 통로가 한 주
하나·닫힌 주만, 자기 이야기 배제가 두 통로에 다 걸리는지) 통과 후 커밋.

## 3. C1 — 출발점 평가와 측정 설정
- 평가: `bash loop/eval4.sh <impl> <run>`을 tmux `bv4_eval`에서(§6). **GEN_RUNS=5**(대화당 생성 5번, 3차·플러그인은 3번),
  JUDGE_REPS=3. 생성 쪽 흔들림이 주 잡음이라 5번으로 늘린다(평균은 3번과 같은 것을 재므로 이전 값과 비교 가능, 덜 흔들릴 뿐).
- 출발점을 **두 번**: `l4_base`, `l4_base_r2`. 채점 `python3 autoresearch/score_2.py <run>`(칸 지킴 조건 + 내용 검사:
  1인칭 복사 ≤ 3, 2인칭 붙여넣기 ≤ 20, 최종 노트 ≤ 250단어, 4–9문장). 비교 대상: L6 두 번 평균 0.1145(기대값 약 0.131),
  플러그인 최종 두 번 평균 0.137.
- 판단: 출발점이 두 번 모두 지킴 조건·내용 검사 통과면 그 두 번 평균이 **현재 최고**. 통과 못 해도 루프는 이 출발점에서 시작하되,
  첫 회차 목표를 실패한 칸으로 둔다. ATTENTION에 결과를 적는다.

## 4. L — 개선 회차
- **고칠 파일은 `belief_update_v7.py` 하나**(+ 그 테스트). 구조까지 바꿔도 된다.
- 한 회차: ① 직전 결과에서 가장 나쁜 칸·흔들림이 큰 대화의 노트와 대화를 읽고 원인 가설 하나 ② PROGRESS에 가설·바꿀 것·예상 사전 등록
  ③ 수정 → 테스트(유출 테스트 포함) → 커밋 ④ `l4_iter_NN` 평가 ⑤ 채점.
- **유지 조건**(10-15 결정, 재실행 쌍 5개에서 단일 실행 sd ≈ 0.025 → GEN_RUNS 5면 더 작다):
  수정본은 두 번 돈다(`l4_iter_NN`, `l4_iter_NN_confirm`). 첫 실행이 지킴·내용 검사 실패거나 MAE > 현재 최고면 confirm 없이 되돌린다.
  **두 실행 모두 지킴·내용 검사 통과 그리고 두 번 평균 MAE ≤ 현재 최고 − 0.03**이면 유지, 아니면 `git revert`.
- 각 회차마다 **흔들림도 기록**한다: 두 실행의 MAE 차이, 대화별 런 간 sd(5런), 칸별 Δ. 남은 오차의 대부분이 흔들림과 대화별 편향이라
  (3차 이어가기 결론) 흔들림을 줄이는 수정도 가치가 있다 — 두 번 평균이 문턱을 못 넘어도 런 간 sd가 크게 줄면 PROGRESS에 적고
  다음 회차 판단에 쓴다(유지 판정 자체는 위 규칙).
- 금지: 개발 대화의 표현·사람·사건을 프롬프트에, 유형·목표·방향을 알려주는 지시, 한쪽으로 미는 지시(대칭 규칙만), 상대 경험의 전유.
- 이미 해 본 것(다시 하지 말 것, 하려면 이유를 적고): call 2 온도만 낮추기(L9), call 1 3표 과반(L10), 노트 코드 조립(L7 — 움직임이 줄어듦),
  보폭 상한·세션당 문장 수 상한(플러그인 5–7회차), 내용 운반 평서문 강제(3차 L3 — 전유 증가), 제자리 교체로 노트 길어짐(3차 L5).
- 종료: 10회차, 또는 연속 5회차 개선 없음, 또는 72시간.

## 5. F — 끝
`results/loop4/REPORT.md`(한국어, 결론 먼저): 출발점 vs 최종 MAE·칸별 Δ·런 간 sd, 회차 표, 바꾼 것과 대칭성 근거, 노트 예시,
L6·플러그인 최종과의 비교, 남은 문제. 진행 그림 `results/loop4/loop4_progress.png`(`results/loop3/loop3_progress_ko.py`와 같은 형식:
제목 "Autoresearch Loss: Belief Update V2", 세로축 "MAE (낮을수록 좋음)", 남색 채택·회색 되돌림·빈 원 방향 조건 실패·옅은 점 재실행 값).
[removed: team coordination]

## 6. GPU·실행
- `loop/eval4.sh <impl> <run>`: 5분마다 빈 GPU를 찾아(<server>·<server> 빈 카드, <server> 빈 카드 **쌍** → 포트 11908) 풀을 띄우고
  `GEN_RUNS=5`로 `loop/eval_v3.sh`를 돈다. **<server>는 쓰지 않는다**(10-15 사용자). 남이 쓰는 카드는 건너뛴다.
- 실행은 항상 `tmux new -d -s bv4_eval "cd <repo>/simulation/belief_v2 && bash loop/eval4.sh <impl> <run> > results/v3_runs/_<run>.log 2>&1"`,
  PROGRESS에 `WAIT_TMUX: bv4_eval`. 평가 하나 약 40–70분(서버 수에 따라).
- 끝(F)에서 `cd simulation/belief_v2 && source loop/gpu_pool.sh && pool_down`.

## 7. 금지·기록
- 시험 세트 금지(§ 위). `harness.py`, `score_v3.py`, `judge_notes.py`, `testset_v3/`, `autoresearch/score_2.py`, `loop/eval_v3.sh`,
  `loop/gpu_pool.sh`, `final/`(새 파일 추가만), 다른 `belief_update_*.py`, `main_experiment/`, `RQ3_crowd/`, 다른 세션의 미커밋 파일은 수정 금지.
[removed: team coordination]
- `results/loop4/ATTENTION.md`에 `## <날짜 UTC> <제목>`: C1 결과, 처음 유지, 이상한 결과(잡음 넘는 퇴보·유지 칸 표류·전유 재발), 큰 구조 변경, F.
