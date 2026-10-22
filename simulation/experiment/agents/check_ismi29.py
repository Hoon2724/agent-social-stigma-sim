"""Validate ISMI-29 measurement files before they reach the export/fit step.

Why this exists (fix-loop S6, 2026-09-28 19:00): the paper judge script
(`paper_rq1/measure_ismi29_run.py`) records a session whose every rep failed
to parse as ``{"n_ok": 0, "n_failed": 3, "error": "no usable measurement"}``
and still exits 0, so ``cluster/measure_arm.sh`` counted the trajectory as
measured. `export_beliefonly_ismi_long.py` then
raises KeyError on that row and the whole arm is dropped from the fit
(fix1/peer_only, level_250 run_01 session 4 -- judge omitted item 25 in all
three reps). This check makes the measurement job fail loudly instead, so a
re-run of the same job re-measures the trajectory.

Usage: ``python check_ismi29.py <file.ismi29.json> [...]`` -- exit 0 when every
file has a numeric ``ismi29_total_mean`` for the pre row and every post row,
1 otherwise (each problem is printed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KEY = "ismi29_total_mean"


def problems(data: dict) -> list[str]:
    """Return a human-readable list of missing measurements in one file."""
    out: list[str] = []
    pre = data.get("ismi29_pre")
    if not isinstance(pre, dict) or not isinstance(pre.get(KEY), (int, float)):
        out.append("pre: no %s" % KEY)
    post = data.get("ismi29_post_by_session")
    if not isinstance(post, list) or not post:
        out.append("post: no sessions")
        return out
    for row in post:
        if not isinstance(row.get(KEY), (int, float)):
            out.append("post session %s: %s (n_ok=%s, n_failed=%s)" % (
                row.get("session"), row.get("error", "no %s" % KEY),
                row.get("n_ok"), row.get("n_failed")))
    return out


def check_file(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return ["unreadable: %s" % error]
    if not isinstance(data, dict):
        return ["not a JSON object"]
    return problems(data)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    bad = 0
    for arg in argv:
        found = check_file(Path(arg))
        for line in found:
            print("%s: %s" % (arg, line))
        bad += bool(found)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
