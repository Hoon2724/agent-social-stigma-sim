# The 1:1 experiment

This folder holds the code and records behind every number in
*What Is Agent-Only Social Media For? A Self-Stigma Intervention as a First Test* (SocialAgent @ NeurIPS 2026).
It covers the one-to-one design only: a Patient agent answered by a Peer agent or by a General agent, five personas at
three initial severity levels, three runs per condition, eight sessions each.

The belief update is `agents/belief_update.py` (sha256 `d295a740…cf037f6e`; Section 3.2.1 and Appendices E–F of
the paper; Appendix F also describes the earlier update it replaced). The persona profiles, the session-0 belief texts and the Peer personas are the cached
texts the reported runs loaded (`agents/config/`, `agents/outputs_v2/`); the scripts that generated them are
`agents/generate_persona_shared.py` and `generate_persona_stigma.py` (GPT-4o; need `OPENAI_API_KEY`) and
`agents/make_session0_belief.py` (Qwen3-14B), but the caches, not a regeneration, are what the runs used. Only the five personas of the paper are included.

## Layout

```
agents/       the agents, the belief update, and the ISMI-29 judge (Python; flat imports, run from this folder)
  prompts/    prompt templates (Appendix B reproduces them)
  config/     experiment settings; session-0 belief texts and Peer personas exactly as loaded by the reported runs
  outputs_v2/ personas file (style examples removed, see STYLE_EXAMPLES.md) and the style-example manifest
scripts/      run_cell.sh (one persona x severity cell), measure.sh (ISMI judge + long table), slurm_cell.sbatch
results/      runs/{peer,general}/level_<L>/run_0k.json   90 run records (dialogue, belief state per session)
              runs/.../run_0k.beliefonly.qwen.ismi29.json   judge records (3 calls per session: totals and subscale means)
              ismi_long.csv                                 one row per trajectory x session
              summaries/                                    numbers behind the tables and figures
              validity/   belief-update runs and accumulated-dialogue scores behind Section 4.2 and Appendix D.2
                          (development dialogues; the sealed split appears only as numbers)
analysis/     Table 3, Figures 2-3 and the Appendix G analyses from results/
evalset/      the 16 development dialogues, the harness and scorer, the scripts that generated and checked the
              evaluation set with Claude (Appendix E), and the sealed split's scores only (sealed_scores/)
development/  how the belief update was developed: the instructions, launch prompts and iteration logs of the four
              search loops (Appendix F) and the evaluation-set design document
minibook/     replay the run records into a Minibook instance and browse them as platform threads
```

Levels: 110/111/112 Avery (severe/moderate/mild), 240–242 Jordan, 250–252 Riley, 260–262 Ravi, 270–272 Noor.

## Reproduce the tables and figures from the released records (CPU only)

```bash
pip install -r requirements.txt
cd analysis
python3 a3_tables.py --data final        # Table 3 and Appendix G tables -> tables_final/
python3 a3_figures.py --data final       # Figures 2-3 and the persona-level figure -> figures_final/
python3 a2_sensitivity.py              # random slope, AR(1) GEE, normalised change (Appendix G)
python3 a7_checks.py            # persona as the unit, strategy mix, subscales, rewrite sizes (Appendix G)
```

Section 4.2 and Appendix D.2 (Figure 6 and its table): `python3 a4_validity.py` recomputes the development
part from `results/validity/` (the sealed part needs the private test split and is taken from
`results/summaries/a4_validity.json`), then `python3 a4_figure.py` and `python3 a5_appendix_tables.py` draw Figure 6
and write the Appendix D.2 and G.4 tables to `tables_final/`. The development-cell table of Appendix F comes from
`../evalset/score_v3.py` on `results/validity/dev_{earlier,final}_r{1,2}`. Not included: the 1–5 ratings of Appendix F
("a further check", run separately) and the score files of the second sealed comparison (their E values are listed in
`evalset/sealed_scores/README.md`).

`results/summaries/` holds the JSON these scripts write, for comparison. To analyse a rerun instead, change the paths
at the top of each script (`SRC`, `DATA`, `RUNS`, `LONG`) to your `ismi_long.csv` and run directory.

## Rerun the experiment (GPU)

Models are served by [Ollama](https://ollama.com): `qwen3:14b` for dialogue and belief updates, `qwen3:32b` for the
judge (Q4_K_M; the paper used Ollama 0.32.9, one 24–48 GB GPU per cell). Qwen3's thinking mode is off
(`think: false`), and conversation memory embeds turns with `sentence-transformers/all-MiniLM-L6-v2`, which is
downloaded from Hugging Face on first use.

1. Put the style examples back: download twelve CSVs of the Reddit Mental Health Dataset (Zenodo 3941387) and run
   `agents/fill_style_examples.py` (`agents/STYLE_EXAMPLES.md`; they are public Reddit posts we do not redistribute).
2. `OLLAMA_URL=http://localhost:11434/v1 bash scripts/run_cell.sh 270` runs one cell, both conditions, three runs.
3. `OLLAMA_URL=... bash scripts/measure.sh results/rerun` scores every session from the belief state alone (judge
   context 4,096 tokens, as in the main runs) and writes `results/rerun/ismi_long.csv`.

The judge averages three calls per belief state; a call with no parsable answers is dropped, so 11 of the 810
released scorings average two calls or one (`n_ok` in each judge record). Some inputs that the final update ignores are
still shipped because the code loads them: `config/stigma_hypothesis/*.txt` and `*.p0.txt`, the `--belief-reps` option,
and `run_dialogues.py` defaults that differ from the paper (always use `scripts/run_cell.sh`).

Run records mark the persona source as `seed_mode: "fabricated"`, meaning a synthetic persona with no seed post. The
records do not store the belief update's hash; the file in `agents/` is the one the runs used (sha256 above).

Python's random module is seeded per run (the seed is in each record); LLM sampling is not, so a rerun reproduces the
distributions, not the exact texts. Settings per step are in Appendix C of the paper and in the commands above.

## Minibook

The reported runs generated each session's thread in process with Minibook's structure: one opening post by the
Patient, then five rounds of replies. `minibook/replay_to_minibook.py` writes those exact threads to a
[Minibook](https://github.com/c4pt0r/minibook) instance (revision `b19c61e`) through its REST API and reads each one
back to check it, so the reported dialogues can be browsed as platform threads. Minibook limits each agent to 10 posts per minute by
default; the script waits and retries, or you can raise the limit in Minibook's `config.yaml`
(`rate_limits: {post: {limit: 1000, window: 60}, comment: {limit: 10000, window: 60}}`).

```bash
cd minibook
python3 setup_minibook.py --base-url http://localhost:3456/api/v1 --project-name paper-replay --out minibook.local.json
python3 replay_to_minibook.py --config minibook.local.json --levels 270 --conditions peer
```

## Evaluation set

`evalset/` holds the sixteen development dialogues, their session-0 texts, and the scorer (Appendix E). The eight
sealed test dialogues are kept private so that they can stay a test set for other belief updates; their per-run
scores for the final update are in `evalset/sealed_scores/`, and the dialogues are available from the authors on
request for research use.

## Data and licences

All agents are simulated and all personas are fictional. The only real-person text is the nine Reddit excerpts used as
style examples (see `agents/STYLE_EXAMPLES.md`; Reddit Mental Health Dataset, Low et al. 2020, ODC-PDDL 1.0), which
are not included. Qwen3 is used under Apache 2.0 and Ollama under MIT; Minibook is a separate project and is not
vendored. This folder is released under the MIT License (`LICENSE`), except the ISMI-29 item text in
`agents/ismi29_items.py`, which belongs to its authors (Ritsher, Otilingam, and Grajales, 2003) and is included only so
the judge can be rerun.
