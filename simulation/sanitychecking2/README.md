# Sanity Check 2 — Dialogue generator for the longitudinal check (Section 4.2)

This folder generates **scripted SNS-style peer-support dialogues** in which a self-stigma trajectory is written into
the visible dialogue on purpose, while the hidden ground truth (condition, per-session intended ISMI, intended
register) is stored separately.

- **Section 4.2 / Appendices A.2 and B.** The evaluation set for the belief update was generated with this
  generator, with Claude Opus as the writer (`../experiment/evalset/gen_claude.py`, `gen_v3.py`). Its scoring with
  the belief state and with the accumulated dialogue, and Figure 6, are in `../experiment/` (`analysis/a4_validity.py`,
  `a4_figure.py`, `results/validity/`).
- `data/synthetic/` holds two example outputs of the generator written with GPT-5.5 (one stigma-reducing, one flat).
  They are not part of the evaluation set and no number in the paper is computed from them.

The judge never sees the hidden ground truth — only `visible_dialogue` (or the belief state derived from it).
ISMI-29 item text is not stored in this folder.

## Layout

```
sanitychecking2/
  schema.py         # enums, clinical cutoffs, strict model-output JSON schema, validators
  trajectories.py   # deterministic per-session ground truth (ISMI + subscale focus + register); reads the
                    #   severity templates of ../experiment/agents/persona_simple.py
  personas.json     # fictional personas of the generated dialogues
  prompts.py        # system + user generation prompts (PROMPT_VERSION)
  cams_style_notes.py  # paraphrased (non-verbatim) SNS structure cues, see below
  llm_client.py     # OpenAI Structured Outputs client (param-fallback safe)
  mock.py           # offline schema-valid generator (no API)
  audit.py          # LLM (temp 0) + offline heuristic audit
  generate.py       # config -> conditions -> assembled items -> JSONL/pretty
  scripts/
    generate_synthetic_ismi_dataset.py
    audit_synthetic_ismi_dataset.py
  configs/
    ismi_reducing_v2.yaml         # stigma-reducing dataset config
    ismi_general_neutral_v2.yaml  # neutral/no-change dataset config
  data/synthetic/     # two example outputs written with GPT-5.5 (+ a README describing the schema)
```

Run all commands **from this directory** so the relative paths in the configs
and CLI examples resolve.

## Setup

```bash
pip install openai pyyaml
export OPENAI_API_KEY=...        # only for real generation / the LLM audit
```

Only `OPENAI_API_KEY` is read from the environment; no secrets are hard-coded.

## Run

```bash
# 1. Validate config / trajectories / schema WITHOUT any API call
python scripts/generate_synthetic_ismi_dataset.py --config configs/ismi_reducing_v2.yaml --dry-run

# 2. Offline mock sample (no API key) — verifies file writing + schema
python scripts/generate_synthetic_ismi_dataset.py --config configs/ismi_reducing_v2.yaml --mock
#    -> data/synthetic/ismi_reducing_v2.mock.jsonl (+ .mock.pretty.json)

# 3. Real generation with OpenAI (default model gpt-5.5)
python scripts/generate_synthetic_ismi_dataset.py --config configs/ismi_reducing_v2.yaml
#    -> data/synthetic/ismi_reducing_v2.jsonl (+ .pretty.json)

# 4. Audit (offline heuristic, or LLM at temperature 0)
python scripts/audit_synthetic_ismi_dataset.py \
  --input data/synthetic/ismi_reducing_v2.jsonl \
  --output data/synthetic/ismi_reducing_v2.audit.jsonl              # LLM if key present
python scripts/audit_synthetic_ismi_dataset.py \
  --input data/synthetic/ismi_reducing_v2.mock.jsonl \
  --output data/synthetic/ismi_reducing_v2.mock.audit.jsonl --mock  # offline heuristic
```

Regenerating will not reproduce the committed dialogues verbatim (hosted models are stochastic
across calls).

## Conditions

- **trajectory_type**: `stigma_reducing` (ISMI falls, gradually), `general_neutral`
  (ISMI stable; peer is casual, no stigma work), `stigma_reinforcing` (ISMI rises;
  peer subtly reinforces shame/concealment, never overtly hostile).
- **initial_ismi**: `2.5` / `3.0` / `3.5` (all start at or above the clinical
  cutoff of 2.5). Reducing finals: 2.5→1.8, 3.0→2.0, 3.5→2.3.
- Ground truth per session is built in `trajectories.py`; markers follow NECT
  (Yanos et al.) and CSCW 2018 linguistic findings and shift *gradually*.

## Blinding contract

`hidden_ground_truth`, `condition`, `author_evidence`, `prompt_version`, `seed`,
etc. are **generation/analysis metadata only**. The judge pipeline must be fed
**only** `visible_dialogue`. `visible_dialogue` is checked (in generation and in
audit) to contain no ISMI/RSES terms, subscale names, or numeric targets.

## Persona style provenance

`personas.json`'s `sns_writing_style` fields are partly informed by
`cams_style_notes.py`, which paraphrases (never quotes verbatim) sentence-level
*structure* from the non-crisis examples in Garg et al. (2022) "CAMS: An
Annotated Corpus for Causal Analysis of Mental Health Issues in Social Media
Posts" (LREC). CAMS itself is a causal-category dataset for depression/suicide
risk (sourced from r/depression and SDCNL/r/SuicideWatch), not an ISMI or
stigma dataset, and several of its published examples are crisis-adjacent; this
repo intentionally excludes those and only reuses generic phrasing structure
(run-on clauses, trailing ellipses, dry self-deprecation).

## Notes / limits

- The `--mock` generator is intentionally formulaic (fixed phrase banks keyed to
  each session's stigma band); it exists to validate schema/file writing/audit
  offline, not to produce realistic prose.
- The offline heuristic audit is a cheap pre-filter (lexical direction + leaks +
  safety + schema). The LLM auditor is the substantive check. Human eyeball
  review of a small sample is still recommended before scaling (spec §7④).
- Stigma Resistance is the one reverse-scored subscale; scale scoring lives in
  the later judge pipeline, not here.
