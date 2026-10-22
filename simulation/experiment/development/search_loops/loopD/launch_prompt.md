> Redacted copy of the authors' working document: lines about team messaging and notifications removed; names, hosts and paths replaced.

Read simulation/belief_v2/autoresearch/instructions_min.md to understand this autoresearch setup.
Work from <repo> (repository root); evals run from simulation/belief_v2.

1. Read simulation/belief_v2/belief_update_v8.py (the minimal starting point) and skim simulation/belief_v2/harness.py
   (generate_run) to see how it is called.
2. The baseline is already measured (2 runs of the starting file): see the config line of
   simulation/belief_v2/autoresearch/autoresearch_min.jsonl and results/v3_runs/arm_base*/_score.md. Do not re-run it.
3. Run the loop exactly as instructions_min.md says (one hypothesis per iteration after the first, eval_min.sh
   start/wait, guard cells AND content, candidate -> confirmation run -> keep if two-run mean >= best + 0.03, else
   revert). Run `cd simulation/belief_v2 && python3 -m pytest -q tests/test_v8_leak.py` before every eval.
   Log every iteration to autoresearch_min.jsonl, regenerate autoresearch_dashboard_min.md, commit both each iteration.
4. 25 iterations (raised from 15 by the user on 10-16; early stop after 6 in a row without a keep). Then the final summary, {"type": "done"}, commit, stop.

IMPORTANT: if you lose context or are resuming, read autoresearch_min.jsonl and autoresearch_dashboard_min.md first
and continue from the last iteration; if an eval was started but not logged, `bash autoresearch/eval_min.sh wait <run>`.
Never open anything under ~/belief_v2_sealed/. Do not read earlier implementations, reports or plans (see the
instructions' "cannot read" list).
If the constrained file has uncommitted changes and an arm_iter_NN eval is running or finished but not logged, that
change IS iteration NN (the previous session was restarted to raise the budget): read it with `git diff`, wait for the
eval, and log/judge it as usual.
