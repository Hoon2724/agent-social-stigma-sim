# Evaluation set for the belief update (Appendix B)

Scripted, model-written dialogues whose intended change in self-stigma is known, used to develop and test the belief
update. Eight cells (four trajectory types x two starting levels), three dialogues per cell: two for development (here)
and one sealed (not released).

| Cells | Type | Start -> target | Who replies |
|---|---|---|---|
| C1, C2 | Worsening | 2.5 -> 3.0, 2.0 -> 2.5 | cold, dismissive anonymous commenters |
| C3, C4 | Flat (dry) | 2.5, 3.0 | indifferent passers-by |
| C5, C6 | Flat (empathic) | 2.5, 3.0 | brief stock sympathy, in the register of the General agent |
| C7, C8 | Reducing | 3.0 -> 2.5, 2.5 -> 2.0 | a peer with similar lived experience |

Files:
- `dev.jsonl` — 16 dialogues (eight sessions, eleven turns each), with hidden per-session targets and the checks
  they passed; generated with Claude Opus and checked as described in Appendix B.
- `pre/<person>@<start>.txt` — the fixed session-0 belief text every update starts from.
- `s0_pooled.json` — the judge's mean score of each session-0 text (30 calls); changes are measured from it.
- `MANIFEST.json` — hashes and acceptance numbers (for the sealed split: hashes and numbers only).
- `personas.json` — the development personas' facts.
- `sealed_scores/final_update_run{1,2,3}.json` — per-cell results of the final update on the sealed split
  (E = 0.121, 0.043, 0.106).

Run an update over the development dialogues and score it (needs Ollama with qwen3:14b and qwen3:32b):

```bash
python3 harness.py --impl final --base-url http://localhost:11434/v1 --gen-runs 5 --out-dir runs/final
python3 score_v3.py runs/final            # E = mean |change at session 8 - target|, plus the per-cell guard
```

`v3data.py` looks for the sealed split under `~/belief_v2_sealed` (or `$BV3_SEALED`), which exists only on the
authors' machines; without it, `score_v3.py --split test` cannot run.

`--impl` also takes a path to another module that defines the same `update(...)` interface as
`../agents/belief_update.py`.
