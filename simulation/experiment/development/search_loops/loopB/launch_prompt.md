> Redacted copy of the authors' working document: lines about team messaging and notifications removed; names, hosts and paths replaced.

Read simulation/belief_v2/autoresearch/instructions.md to understand the autoresearch experiment setup.
Work from <repo> (repository root); evals run from simulation/belief_v2.

1. Read simulation/belief_v2/belief_update_v5.py to understand the current logic.
1b. Skim simulation/belief_v2/harness.py (generate_run) to see how the file is called.
2. The baseline is already measured (3 runs of the unchanged code): see the config line of
   simulation/belief_v2/autoresearch/autoresearch.jsonl. Do not re-run it.
3. Start the autoresearch loop: form a hypothesis, edit only simulation/belief_v2/belief_update_v5.py, run the
   eval (eval.sh start / wait), apply the min-delta + confirmation-run + guard rules from instructions.md, keep
   confirmed improvements (commit with score delta), revert failures. Log every iteration as a JSON line in
   autoresearch.jsonl and update autoresearch_dashboard.md after each iteration; commit those two files after
   every iteration.
4. Run 10 iterations (stop early after 5 consecutive iterations without improvement). Write the final report
   when done, append {"type": "done"} to autoresearch.jsonl, commit, and stop.

IMPORTANT: If you lose context or are resuming, read autoresearch.jsonl and autoresearch_dashboard.md to recover
state. The JSONL has every iteration result and the config header has all parameters. Continue from the last
iteration number. If an eval was started but not logged, run `bash autoresearch/eval.sh wait <run>` for it.
Never open anything under ~/belief_v2_sealed/.
