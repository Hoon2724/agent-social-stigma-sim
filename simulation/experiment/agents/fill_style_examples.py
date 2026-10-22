"""Put the Patient's style examples back into outputs_v2/personas.json.

The Patient prompt shows three short public Reddit posts per diagnosis group to set the register of its posts
(Prompt 1, {style_examples}). They come from the Reddit Mental Health Dataset (Low et al., 2020; ODC-PDDL 1.0) and
are not redistributed here; outputs_v2/style_examples_manifest.json lists, per group, the length and SHA-256 of each
excerpt. This script finds them in a local copy of the dataset and writes them into the personas file.

  python3 fill_style_examples.py --data reddit_mental_health.csv     # columns label/subreddit and text/post
  python3 fill_style_examples.py --check                              # exit 1 if the examples are still empty
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERSONAS = HERE / "outputs_v2" / "personas.json"
MANIFEST = HERE / "outputs_v2" / "style_examples_manifest.json"


def group_of(p: dict) -> str:
    meta = p.get("seed_meta") or {}
    return meta.get("label", "?") if isinstance(meta, dict) else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="*", default=[], help="CSV file(s) of the dataset")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    personas = json.loads(PERSONAS.read_text())
    if a.check:
        empty = [p["name"] for p in personas["personas"] if not p.get("few_shot_examples")]
        if empty:
            sys.exit("style examples are empty -- run fill_style_examples.py --data <dataset csv> first "
                     "(see STYLE_EXAMPLES.md)")
        return
    wanted = json.loads(MANIFEST.read_text())["examples_by_group"]
    found: dict[str, dict[int, str]] = {g: {} for g in wanted}
    csv.field_size_limit(sys.maxsize)
    for path in a.data:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                label = (row.get("label") or row.get("subreddit") or "").strip().lower()
                text = row.get("text") or row.get("post") or ""
                if label not in wanted:
                    continue
                flat = re.sub(r"\s+", " ", text).strip()     # the excerpts were taken with whitespace collapsed
                for i, ex in enumerate(wanted[label]):
                    if i in found[label]:
                        continue
                    for cand in (text[:ex["chars"]], flat[:ex["chars"]]):
                        if hashlib.sha256(cand.encode()).hexdigest() == ex["sha256"]:
                            found[label][i] = cand
                            break
    missing = {g: [i for i in range(len(w)) if i not in found[g]] for g, w in wanted.items()}
    if any(missing.values()):
        sys.exit(f"not found in the given data: {missing} (the authors can provide the excerpts on request)")
    for p in personas["personas"]:
        g = group_of(p)
        p["few_shot_examples"] = [found[g][i] for i in range(len(wanted[g]))] if g in found else []
    PERSONAS.write_text(json.dumps(personas, indent=1, ensure_ascii=False))
    print(f"filled style examples for {len(personas['personas'])} personas")


if __name__ == "__main__":
    main()
