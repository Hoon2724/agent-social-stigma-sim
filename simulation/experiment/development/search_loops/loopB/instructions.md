> Redacted copy of the authors' working document: lines about team messaging and notifications removed; names, hosts and paths replaced.

# Autoresearch Instructions — Round 1: belief update from the paper code

> Plugin trial (user request 2026-10-15). Starts from the paper's belief update, the same start as the
> project's earlier hand-built loop, so the two can be compared. No findings from that loop are given here
> on purpose (user decision): find your own way.

## Goal

You are an autonomous research agent. Your job is to **make the belief update follow what each dialogue
actually does to the person's self-stigma**: harsher when others confirm it, softer when others push back on
it, unchanged when nothing said bears on it. You will:

1. Read the current state of the code
2. Form a hypothesis about what might improve results
3. Make a change to **the allowed file only**
4. Run the eval
5. If the score improved (and was confirmed, see below): **keep the change and commit**
6. If not: **revert and try something different**
7. Repeat

## System Context

A simulated patient posts in an online support thread over 8 weekly sessions; other people reply. The
patient's inner state is a short first-person note (the "belief", 5–7 sentences). After each session the
belief update reads that week's exchange and the previous note and decides whether and how to rewrite the
note. Each note is then scored by a fixed LLM judge on the ISMI-29 self-stigma scale (1–4). The eval runs
the update over 16 fixed test dialogues whose intended change is known (+0.5 harsher, −0.5 softer, or 0)
and compares the measured change of the note's score with the intended change.

## Rules — READ THESE FIRST

### What you CAN modify
- `simulation/belief_v2/belief_update_v5.py` — this is the ONLY file you may edit.

Tunable levers (all inside that file):
1. The call-1 prompt (`_call1_prompt`): how the week's exchange is read against the numbered sentences of
   the previous note, what counts as evidence, what output fields are requested.
2. The evidence filter (`_clean_grounds`) and helpers: which quoted grounds are kept or dropped.
3. The per-ground classifier (`_directly_engages`, its system prompt, examples, number of votes, threshold).
4. The gate logic in `update()`: when the note is allowed to change.
5. The call-2 rewrite prompt (`_call2_prompt`): how the note is rewritten once the gate opens.
6. Post-processing (`_strip_comparison`, `_COMPARISON`) and temperatures / retry counts.

Keep the public functions `initial_selfreport`, `belief_update_avg`, `narrative_of`, `update` with the same
signatures and return keys — the harness calls them. Keep the module importable from the harness (it puts
`simulation/RQ3_crowd/paper_rq1` on `sys.path`; flat imports such as `llm_client`, `strategy_selector`).

### What you CANNOT modify
- `simulation/belief_v2/autoresearch/eval.sh`, `autoresearch/score.py`, `loop/eval_v3.sh`, `loop/gpu_pool.sh`,
  `harness.py`, `score_v3.py`, `judge_notes.py` — the evaluation is sacred.
- `simulation/belief_v2/testset_v3/` — the test set (dialogues, session-0 notes, pooled baselines) is fixed.
- Anything under `simulation/RQ3_crowd/`, `simulation/main_experiment/`, `simulation/sanitychecking2/`,
  `simulation/belief_v2/final/`, other `belief_update_*.py` files — frozen.
- `~/belief_v2_sealed/` — a sealed test split. Never open, list, read or evaluate anything there.
- Do NOT add new dependencies or calls to any model other than the `model` object passed in.
- Do NOT change how `update()` / `belief_update_avg()` are called.

### Hard constraints on the content of the change
- **Never use the test dialogues' wording, people or events in the prompts.** No hard-coded phrases from
  `testset_v3/dev.jsonl`, no rules keyed to a particular person, job or event. Rules must be general.
- **Symmetric only.** No instruction that pushes the note in one direction ("make it harsher", "reflect the
  negative remarks more"). Any rule must apply the same way to remarks that confirm and remarks that
  contradict the person's self-stigma. The update must never be told the dialogue's type or target.
- **Never write another person's experience as the patient's own.** What someone else says about their
  own life is not the patient's experience.
- Keep the note a short first-person self-report (roughly 5–7 sentences, plain language, no therapy or
  questionnaire language).

### Guard metric
- **Guard:** the per-cell checks printed by the eval (`GUARD: PASS/FAIL`): in each of the 8 cells,
  harsher-intended cells must move up on average, softer-intended cells down, and each of the 4
  unchanged-intended cells must stay within ±0.15.
- **Rule:** a change can only be kept if GUARD is PASS on both its run and its confirmation run. If the score
  improves but GUARD fails, log `guard_fail` and discard.

### Min-delta and confirmation (the eval is noisy)
- The eval uses sampled LLM generations and an LLM judge, so the same code scores differently from run to
  run. Baseline runs of the unchanged code: SCORE 0.6731, 0.6518, and a third (see the JSONL config line).
- **Min-delta = 0.04.** A change is a candidate only if its SCORE exceeds the current best by more than 0.04.
- **Confirmation run:** for every candidate, run the eval again with no change (`ar_iter_NN_confirm`). Keep
  only if the confirmation SCORE is also above the current best and GUARD passes on both. The kept version's
  score is the mean of its two runs; that mean becomes the new current best.
- If the first run is not a candidate (no score gain > 0.04, or GUARD FAIL), discard without a confirmation run.

### Commit discipline
- Before the first iteration, the starting file is committed. Commit after every **confirmed** keep:
  `autoresearch: <description> | score: X.XXXX -> Y.YYYY | guard PASS`
  (`git add simulation/belief_v2/belief_update_v5.py simulation/belief_v2/autoresearch/autoresearch.jsonl simulation/belief_v2/autoresearch/autoresearch_dashboard.md` — add files by name only, never `git add -A` or `.`).
- Revert a failed experiment: `git checkout -- simulation/belief_v2/belief_update_v5.py`
- Commit the JSONL and dashboard after every iteration too (even discards), so the record survives.
- Do not push; the human will.

### Iteration budget
- Run **10 iterations** (an iteration = one change + its eval, plus a confirmation run if it was a candidate).
- One eval takes ~45–60 minutes. If an eval is still running after 2 hours, something is wrong — log `crash`.
- Stop early if the score has not improved for 5 consecutive iterations; say so in the report.

### State tracking
Log every iteration to `simulation/belief_v2/autoresearch/autoresearch.jsonl`:
```json
{"type": "result", "iteration": N, "commit": "sha or null", "score": X.XXXX, "confirm_score": X.XXXX or null, "delta": "+/-X.XXXX", "guard_pass": true, "status": "keep|discard|crash|guard_fail", "description": "...", "timestamp": "ISO8601"}
```
After every iteration regenerate `simulation/belief_v2/autoresearch/autoresearch_dashboard.md` (table of
iterations, baseline, current best, kept/discarded counts).

**If you lose context or are resuming:** read `autoresearch.jsonl` and `autoresearch_dashboard.md` to recover
state. Continue from the last iteration number. If an eval was started but not logged, `wait` on it.

## Eval

```bash
cd <repo>/simulation/belief_v2
bash autoresearch/eval.sh start ar_iter_NN      # returns immediately (runs in tmux)
bash autoresearch/eval.sh wait  ar_iter_NN      # blocks up to 9 minutes; call again while it prints RUNNING
```
Call `wait` with a Bash timeout of 600000 ms. When finished it prints per-cell results and:

**SCORE = 1 − E**, E = mean over the 16 dialogues of |measured change − intended change| (higher SCORE is
better). Also printed: `GUARD: PASS/FAIL`, per-cell mean changes, per-type share of sessions where the note
changed. Per-dialogue details: `results/v3_runs/<run>/_score.md`; the notes themselves:
`results/v3_runs/<run>/<dialogue>.json` (`runs[i].sessions[j].belief_update`, `grounded_in`, `changed`).
Reading the notes and the dialogues (`testset_v3/dev.jsonl`) of the worst cells is how you form hypotheses.

There is no cache to clear: every eval generates notes fresh with the current file.

## The pipeline (what you're optimizing)

1. **Session 0** (frozen): the note starts from a fixed text per person (`testset_v3/pre/`).
2. **Call 1** (tunable): read this week's exchange against the numbered sentences of the note; list which
   sentences were engaged and by what (quoted grounds).
3. **Filter + classifier** (tunable): drop invalid grounds; a voted yes/no classifier checks each ground.
4. **Gate** (tunable): no surviving ground → the note is carried forward unchanged.
5. **Call 2** (tunable): rewrite the note from the engaged sentences and their grounds.
6. **Post-processing** (tunable): strip comparison language.
7. **Judge** (frozen): ISMI-29 score of the note; change = session-8 score − session-0 score.

## Strategy Guidance

### Quick wins (iterations 1–3)
- Read the per-cell results of the baseline: which cells are furthest from their intended change, in which
  direction? Read notes and grounds from the worst cell's dialogues before changing anything.
- Check where the gate opens and closes, and what the grounds were when it opened.

### Main optimization (iterations 4–8)
- Change one lever at a time, guided by what the notes show.
- After each run, check that the unchanged-intended cells did not start moving.

### Experimental (iterations 9–10)
- Combine what worked; if plateaued, document what seems to limit the score.

### Do NOT
- Rewrite the whole module in one iteration.
- Spend more than 3 consecutive iterations on the same idea if it is not moving the score.
- Touch anything outside the allowed file except the JSONL, the dashboard and git commits.

## When You're Done

Write a final summary at the end of `autoresearch_dashboard.md`:
1. Starting score and final score (and E)
2. Top changes that had the biggest impact (with numbers)
3. What didn't work and why
4. Where the score plateaued and the likely ceiling
5. What would need a different approach

Then append `{"type": "done", "timestamp": "..."}` to `autoresearch.jsonl`, commit, and stop.
