# Sanity Check 1 — Static Measurement Validity

Reproduces Section 4.1 and Figure 5 (Appendix A.1): does an ISMI-29 LLM judge
recover a predefined ordering of self-stigma severity from persona text it was
not told the levels of?

Ten variants of one Patient persona (Avery) are generated, differing only in
self-stigma level, then scored blind by the judge. The reported result is
Spearman's rho = 0.99 between target level and measured ISMI-29, with one
adjacent inversion (levels 2--3, by 0.04, within the within-level SD).

## Layout

```
generate_avery_10level.py   # GPT-4o: writes the 10 persona levels
measure_single_level.py     # Qwen3-32B: scores ONE level, N judge repetitions
run_measure_levels.sh       # job queue: all 10 levels across N ollama endpoints
make_figure.py              # renders Figure 5 from the measurements
outputs/
  avery_10level_personas.json      # the 10 generated levels (committed)
  level_ismi29_30reps/level_*.json # per-level scores, 30 reps each (committed)
  figures/avery_10level_ismi29.pdf # Figure 5
```

Committed outputs are the exact ones behind the reported numbers, so the figure
can be regenerated without re-running generation or measurement.

## How levels are specified

Levels are defined by structural coordinates (`_LEVEL_COORDINATES` in
`generate_avery_10level.py`), not by severity adjectives. Each level fixes a
position on the three axes the main simulation's own persona generation varies:
what the condition has come to mean about the person, how much each of the three
close relationships knows, and what they can say out loud. The generation model
renders these as prose; it never sees the ISMI items, a numeric target, or a
severity label.

This matters for reproduction: asking the model for "ten evenly spaced levels"
with only the ordinal to go on does not work (we measured rho = 0.21 — the model
varies intensity adverbs instead of the situation, and the middle of the range
collapses into one band).

## Dependencies

Python: `openai`, `matplotlib`. Reads (never writes) four modules of the main simulation in
`../experiment/agents/`: `reddit_persona_profiles`, `measure_ismi29` (the ISMI-29 judge of Appendix E, Prompt 11),
`ismi29_items`, `llm_client`. These are the modules the reported measurement used.

Serving: Qwen3-32B via ollama (Q4_K_M, ~19GB, fits one 24GB GPU).

```bash
export OPENAI_API_KEY=...     # or place it in ../.env (auto-loaded)
```

Only step 2 (generation) needs the key. Measuring and plotting the committed
outputs need only ollama and matplotlib.

## Run

```bash
# 1. Inspect the prompt without calling the API
python3 generate_avery_10level.py --dry-run

# 2. Generate the 10 levels (GPT-4o, temperature 0)
#    Overwrites outputs/avery_10level_personas.json
python3 generate_avery_10level.py

# 3. Score every level, 30 judge repetitions each.
#    Pass one or more ollama endpoints; levels are distributed across them.
./run_measure_levels.sh localhost:11434
./run_measure_levels.sh host1:11434 host2:11434 host3:11434   # parallel

# ...or score a single level directly
python3 measure_single_level.py \
  --input outputs/avery_10level_personas.json \
  --level level_1 --base-url http://localhost:11434/v1 \
  --reps 30 --out outputs/level_ismi29_30reps/level_1.json

# 4. Render Figure 5
python3 make_figure.py
```

Step 2 is stochastic across API calls even at temperature 0, so regenerating
will not reproduce the committed personas verbatim. To reproduce the reported
numbers exactly, skip step 2 and measure the committed
`outputs/avery_10level_personas.json`.

## Notes

- `run_measure_levels.sh` uses a single shared job queue with a lock, so adding
  endpoints only adds parallelism — no level is measured twice.
- Judge context is 4096: the prompt is one persona text plus the 29 items, well
  under that. Larger contexts are unnecessary here.
- The ISMI-29 item text is not stored in this folder; it is loaded from
  `../experiment/agents/ismi29_items.py`.
