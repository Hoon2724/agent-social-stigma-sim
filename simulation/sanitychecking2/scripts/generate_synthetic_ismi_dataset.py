#!/usr/bin/env python
"""CLI: generate a synthetic ISMI sanity-check dataset from a config.

Examples (run from the module directory
simulation/sanitychecking2/):

    # offline mock sample (no API key needed) -> data/synthetic/<name>.mock.jsonl
    python scripts/generate_synthetic_ismi_dataset.py --config configs/ismi_pilot.yaml --mock

    # validate config/schema without calling the API
    python scripts/generate_synthetic_ismi_dataset.py --config configs/ismi_pilot.yaml --dry-run

    # real generation with OpenAI (needs OPENAI_API_KEY)
    python scripts/generate_synthetic_ismi_dataset.py --config configs/ismi_pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the module importable when invoked as scripts/<file>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a YAML/JSON config")
    ap.add_argument("--mock", action="store_true", help="offline generation, no API")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config/trajectories/schema without generating",
    )
    ap.add_argument("--model", help="override the generation model name")
    ap.add_argument("--limit", type=int, help="cap number of samples (smoke test)")
    args = ap.parse_args()

    config = generate.load_config(args.config)
    summary = generate.run(
        config,
        mock=args.mock,
        dry_run=args.dry_run,
        limit=args.limit,
        model_override=args.model,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    problems = summary.get("problems") or summary.get("dry_run_problems")
    if problems:
        print(f"\nWARNING: {len(problems)} sample(s) had schema/leak problems.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
