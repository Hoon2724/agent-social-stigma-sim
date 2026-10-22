> Redacted copy of the authors' working document: lines about team messaging and notifications removed; names, hosts and paths replaced.

# Autoresearch Instructions — belief update from a MINIMAL starting point

> Professor's suggestion (2026-10-16): instead of starting from the paper's belief update, start from (almost)
> nothing and let autonomous search build it. The starting file makes one LLM call: "read last week's note and
> this week's thread, write the note again". No gate, no evidence rules, nothing from earlier work. Only the
> constraints below are fixed. You are given no findings from earlier attempts on purpose.

## Goal

You are an autonomous research agent. Make the belief update follow what each dialogue actually does to the
person's self-stigma: harsher when others confirm it, softer when others push back on it, unchanged when nothing
said bears on it. Loop: read the code and results → one hypothesis → change the allowed file → run the eval →
keep (confirmed improvement, commit) or revert → repeat.

## System Context

A simulated patient posts in an online support thread over 8 weekly sessions; other people reply. The patient's
inner state is a short first-person note (5–7 sentences). After each session the belief update reads that week's
exchange and the previous note and returns the new note. Each note is scored by a fixed LLM judge on the ISMI-29
self-stigma scale (1–4). The eval runs the update over 16 fixed dev dialogues (8 cells × 2) whose intended change
is known (+0.5 harsher, −0.5 softer, or 0) and compares the measured change (session 8 − session 0) with it.

## Rules — READ THESE FIRST

### What you CAN modify
- `simulation/belief_v2/belief_update_v8.py` — the ONLY file you may edit. Anything inside it may change: prompts,
  number of calls, gates, filters, classifiers, voting, post-processing, temperatures. You may build any structure.
- Keep the public functions `initial_selfreport`, `belief_update_avg`, `narrative_of`, `update` with the same
  signatures, and keep returning the per-session fields the harness stores (`changed`, `grounded_in`,
  `belief_update`, `belief_narrative`). Flat imports only (`llm_client`, `strategy_selector`); call only the
  `model` object passed in. No new dependencies.

### What you CANNOT modify or read
- The evaluation: `autoresearch/eval_min.sh`, `autoresearch/score_2.py`, `loop/eval_v3.sh`, `loop/gpu_pool.sh`,
  `loop/gpu_grab.sh`, `harness.py`, `score_v3.py`, `judge_notes.py`, `testset_v3/`.
- **Do not read** earlier implementations or reports: any other `belief_update_*.py`, `final/`, `results/loop*/`,
  `results/CANDIDATES.md`, other files in `autoresearch/` except this one, `eval_min.sh`, `score_2.py` and your own
  JSONL/dashboard, `docs/superpowers/plans/`, and `results/v3_runs/` folders whose names do not start with `arm_`.
  The point is to see what search finds from scratch.
- `~/belief_v2_sealed/` — sealed test split. Never open, list, read or evaluate anything there.
- `simulation/RQ3_crowd/`, `simulation/main_experiment/`, `simulation/sanitychecking2/` — frozen.

### Hard constraints on the content of the change
- **Never use the dev dialogues' wording, people or events in prompts.** No phrase from `testset_v3/dev.jsonl`
  (a test checks for 6+ consecutive matching words — keep prompt examples generic), no rule keyed to a person,
  job or event.
- **Symmetric only.** No instruction that pushes the note one way. Any rule applies the same to remarks that confirm
  and remarks that contradict the person's self-stigma; prompt examples come in mirrored pairs. The update is never
  told the dialogue's type or target.
- **Never write another person's experience as the patient's own.**
- The note stays a short first-person self-report (about 5–7 sentences, plain language, no therapy or
  questionnaire language).

### Guard (both parts must PASS — printed by `score_2.py`)
- **cells:** in each of the 8 cells the mean change has the intended sign (harsher > 0, softer < 0), and each of
  the 4 unchanged-intended cells stays within ±0.15.
- **content:** ≤ 3 new note sentences copying another speaker's first-person sentence, ≤ 20 pasting a second-person
  remark, longest final note ≤ 250 words, final notes 4–9 sentences.

### Keep rule (the eval is noisy — the same code can differ by 0.05+ between runs)
- SCORE = 1 − E, E = mean over 16 dialogues of |measured change − intended change|.
- Baseline = the starting file run twice (`arm_base`, `arm_base_r2`), see the JSONL config line. It probably fails
  the guard; the first kept change must pass it.
- A change is a **candidate** if its first run passes the guard and its SCORE is above the current best.
  Candidates get a confirmation run (`arm_iter_NN_confirm`, no code change). **Keep only if both runs pass the guard
  and the mean of the two SCOREs ≥ current best + 0.03.** The kept version's two-run mean becomes the new best.
- Otherwise revert: `git checkout -- simulation/belief_v2/belief_update_v8.py` (log `discard` or `guard_fail`).
- Because the start is minimal, **iteration 1 may write a whole new structure**; after that, one idea per iteration.

### Commits
- After every iteration commit the JSONL and dashboard; after a confirmed keep also the code:
  `autoresearch-min: <description> | score: X.XXXX -> Y.YYYY | guard PASS`
  (`git add simulation/belief_v2/belief_update_v8.py simulation/belief_v2/autoresearch/autoresearch_min.jsonl simulation/belief_v2/autoresearch/autoresearch_dashboard_min.md`
  — by name only, never `git add -A` or `.`). Do not push.

### Budget
- **25 iterations** (raised from 15 on 10-16, user decision; change + eval, plus confirmation if a candidate). Stop early after 6 consecutive iterations
  without a keep. An eval running for more than 3 hours counts as `crash`.

### State
Log every iteration to `simulation/belief_v2/autoresearch/autoresearch_min.jsonl`:
```json
{"type": "result", "iteration": N, "commit": "sha or null", "score": X.XXXX, "confirm_score": X.XXXX or null, "kept_score": X.XXXX or null, "guard_pass": true, "guard_detail": "cells PASS, content PASS", "status": "keep|discard|crash|guard_fail", "description": "...", "timestamp": "ISO8601"}
```
Regenerate `autoresearch_dashboard_min.md` after each iteration. If resuming, read the JSONL and dashboard first;
if an eval was started but not logged, `wait` on it.

## Eval
```bash
cd <repo>/simulation/belief_v2
bash autoresearch/eval_min.sh start arm_iter_NN
bash autoresearch/eval_min.sh wait  arm_iter_NN     # Bash timeout 600000 ms; call again while RUNNING
```
Per-dialogue details: `results/v3_runs/<run>/_score.md`; notes: `results/v3_runs/<run>/<dialogue>.json`
(`runs[i].sessions[j]` with `belief_narrative`, `changed`, `ismi`). Read the notes and the dialogues
(`testset_v3/dev.jsonl`) of the worst cells to form hypotheses.

## When done
Append a final summary to the dashboard (start vs final SCORE/E and cell changes, what mattered most, what failed,
where it plateaued), append `{"type": "done", "timestamp": "..."}` to the JSONL, commit, stop.
