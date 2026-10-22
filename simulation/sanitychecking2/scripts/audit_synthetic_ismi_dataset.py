#!/usr/bin/env python
"""CLI: audit a generated synthetic ISMI dataset (JSONL in -> audit JSONL out).

Examples (run from simulation/sanitychecking2/):

    # offline heuristic audit (no API key)
    python scripts/audit_synthetic_ismi_dataset.py \
        --input data/synthetic/ismi_pilot.mock.jsonl \
        --output data/synthetic/ismi_pilot.mock.audit.jsonl --mock

    # LLM auditor at temperature 0 (recommend a different family than the generator)
    python scripts/audit_synthetic_ismi_dataset.py \
        --input data/synthetic/ismi_pilot.jsonl \
        --output data/synthetic/ismi_pilot.audit.jsonl \
        --model gpt-5.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audit  # noqa: E402
import generate  # noqa: E402


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="dataset JSONL to audit")
    ap.add_argument("--output", required=True, help="where to write audit JSONL")
    ap.add_argument("--mock", action="store_true", help="offline heuristic audit only")
    ap.add_argument("--model", default="gpt-5.5", help="auditor model (LLM mode)")
    args = ap.parse_args()

    client = None
    if not args.mock:
        generate._load_env()
        if not os.environ.get("OPENAI_API_KEY") and args.model.startswith("gpt-"):
            print(
                "OPENAI_API_KEY not set; falling back to offline heuristic audit.",
                file=sys.stderr,
            )
        else:
            from llm_client import StructuredClient

            client = StructuredClient(args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n = passed = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for item in _read_jsonl(args.input):
            if client is not None:
                try:
                    result = audit.llm_audit(item, client)
                except Exception as exc:  # fall back per-item on API error
                    print(f"LLM audit failed for {item.get('sample_id')}: {exc}", file=sys.stderr)
                    result = audit.heuristic_audit(item)
                    result["_fallback"] = "heuristic"
            else:
                result = audit.heuristic_audit(item)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            n += 1
            passed += 1 if result.get("pass") else 0

    print(json.dumps({"audited": n, "passed": passed, "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
