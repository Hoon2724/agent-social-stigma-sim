# How the evaluation set and the belief update were developed (Appendices E and F)

## Evaluation set (`../evalset/`)

The 24 scripted dialogues were generated with the longitudinal-dialogue generator in
`../../sanitychecking2/generate.py` (prompts in `../../sanitychecking2/prompts.py`), driven by Claude Opus, with
per-session targets reaching a change of +0.5, −0.5 or 0, and an extra instruction for the General-like replies.

| Step | Script (in `../evalset/`) | Model |
|---|---|---|
| Generate a dialogue toward hidden per-session targets | `gen_v3.py` → `gen_claude.py` → sanitychecking2 generator | Claude Opus 5.5 (`claude -p`) |
| Blinded direction check | `check_v3.py direction` (`direction_check.py`) | Claude Sonnet 5.5 |
| Reference belief-state sequences, scored by the Qwen3-32B judge | `check_v3.py oracle` (`oracle_notes.py`) | Claude Opus or Sonnet 5.5 |
| Review by a separate instance | `check_v3.py review` | Claude Sonnet 5.5 |
| Session-0 belief text and its calibration | `s0_calibrate.py` (prompt `PRE_PROMPT` in `harness.py`, registers in `registers.json`) | Qwen3-14B, judge Qwen3-32B |
| Hashes and acceptance numbers | `manifest_v3.py` → `MANIFEST.json` | — |

`evalset_plan.ko.md` is the design document the generation followed (in Korean; redacted copy).

## Search loops (`search_loops/`)

Each loop ran an LLM coding agent (Claude Code) on one belief-update file with a fixed evaluation (`../evalset/harness.py`
+ `score_v3.py`, development split only) and kept an edit only if the error E improved by a preset margin over a run and a
confirmation run while the per-cell guard held (Appendix F). The agent was instructed never to open the sealed split and
was denied tool access to it.

| Loop | Started from | What drove it | Files |
|---|---|---|---|
| A | the earlier update | a shell driver calling `claude -p` once per iteration with a launch prompt and a written instruction document | `loopA/driver.sh`, `launch_prompt.ko.md`, `instructions.ko.md` |
| B | the earlier update | the packaged autoresearch plugin: `instructions.md`, `launch_prompt.md`, eval and score scripts, iteration log | `loopB/` |
| C | Loops A and B combined by hand (the final update) | same driver as A | `loopC/driver.sh`, `launch_prompt.ko.md`, `instructions.ko.md` |
| D (control) | a single LLM call | autoresearch setup with no prior design | `loopD/` |

`loopB/autoresearch.jsonl` and `loopD/autoresearch_min.jsonl` record every iteration (score = 1 − E, status, the change
tried). `a6_search_figure.py` draws Figure 7 of the paper from these logs and the loop A/C progress records.

For Loops A and C, `driver.sh` called `claude -p` with `launch_prompt.ko.md` (named `prompt3.md` / `prompt4.md` in the
driver) and the tool permissions in `agent_settings.json` (which also deny access to the sealed split); the iteration
records are in `iterations.json`. Loops B and D keep their permissions in `agent_settings.json` too.
`[removed: team coordination]` marks where lines about messaging were cut.

The documents for Loops A and C are the authors' working instructions in Korean; lines about team messaging and
notifications were removed and names, hosts and paths replaced. The evaluation scripts they call ran on the authors' GPU
pool (scripts not included); `../evalset/harness.py` and `score_v3.py` are the parts that define the evaluation.
