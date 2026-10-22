#!/usr/bin/env python3
"""Export belief-only ISMI session trajectories to long-format CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LEVEL_META = {
    110: ("Avery", "severe"),
    111: ("Avery", "moderate"),
    112: ("Avery", "mild"),
    240: ("Jordan", "severe"),
    241: ("Jordan", "moderate"),
    242: ("Jordan", "mild"),
    250: ("Riley", "severe"),
    251: ("Riley", "moderate"),
    252: ("Riley", "mild"),
    260: ("Ravi", "severe"),
    261: ("Ravi", "moderate"),
    262: ("Ravi", "mild"),
    270: ("Noor", "severe"),
    271: ("Noor", "moderate"),
    272: ("Noor", "mild"),
}


def iter_rows(root: Path):
    for condition in ("peer", "general"):
        for level_dir in sorted((root / condition).glob("level_*")):
            level = int(level_dir.name.split("_", 1)[1])
            if level not in LEVEL_META:
                continue
            persona, severity = LEVEL_META[level]
            for measure_path in sorted(level_dir.glob("run_*.beliefonly.qwen.ismi29.json")):
                data = json.loads(measure_path.read_text(encoding="utf-8"))
                run_name = measure_path.name.split(".", 1)[0]
                run_idx = int(run_name.split("_", 1)[1])
                trajectory_id = f"{persona}_{severity}_{condition}_{run_name}"
                yield {
                    "persona": persona,
                    "severity": severity,
                    "level": level,
                    "condition": condition,
                    "run": run_name,
                    "run_idx": run_idx,
                    "trajectory_id": trajectory_id,
                    "session": 0,
                    "ismi": data["ismi29_pre"]["ismi29_total_mean"],
                    "ismi_sd": data["ismi29_pre"]["ismi29_total_sd"],
                }
                for session_row in data["ismi29_post_by_session"]:
                    yield {
                        "persona": persona,
                        "severity": severity,
                        "level": level,
                        "condition": condition,
                        "run": run_name,
                        "run_idx": run_idx,
                        "trajectory_id": trajectory_id,
                        "session": int(session_row["session"]),
                        "ismi": session_row["ismi29_total_mean"],
                        "ismi_sd": session_row["ismi29_total_sd"],
                    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = list(iter_rows(args.root))
    if not rows:
        raise SystemExit(f"no rows found under {args.root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "persona",
                "severity",
                "level",
                "condition",
                "run",
                "run_idx",
                "trajectory_id",
                "session",
                "ismi",
                "ismi_sd",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
