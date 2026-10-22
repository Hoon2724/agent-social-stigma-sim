"""measure_single_level.py — administer ISMI-29 to ONE Avery level, for use
by run_measure_pool.sh's job queue (one level per GPU worker, in parallel).
Same rendering/measurement logic as measure_avery_10level.py, split out so
multiple levels can run concurrently across multiple ollama endpoints.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent / "experiment" / "agents"   # the main simulation's agent and judge code
sys.path.insert(0, str(AGENTS))

from ismi29_items import ISMI_SUBSCALES  # noqa: E402
from llm_client import OpenAICompatibleModel  # noqa: E402
from measure_ismi29 import interpret, measure_once  # noqa: E402
from reddit_persona_profiles import _render_meta, _voice, _render_static_profile_longitudinal  # noqa: E402


def build_backstory(meta: dict, profile: dict, level_block: dict) -> str:
    v = _voice(meta)
    full_profile = {**profile, **level_block}
    static_profile = _render_static_profile_longitudinal(full_profile, v["poss"], v["subj"])
    return (
        f"{_render_meta(meta)}\n\n"
        f"Deep down, {v['subj']} believe{v['s']}: \"{profile['core_belief']}\"\n\n"
        f"When things get hard, thoughts like this tend to come up: "
        f"\"{profile['automatic_thought']}\""
        + (f"\n\n{static_profile}" if static_profile else "")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--level", required=True, help="e.g. level_1")
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--base-url")
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    meta, profile = data["meta"], data["profile"]
    level_block = data["levels"][args.level]
    backstory = build_backstory(meta, profile, level_block)

    model = OpenAICompatibleModel(args.model, args.base_url, num_ctx=args.num_ctx)

    results, failures = [], 0
    for i in range(args.reps):
        try:
            results.append(measure_once(model, backstory, args.temperature))
        except (ValueError, KeyError, TypeError) as exc:
            failures += 1
            print(f"  rep {i+1} failed: {exc}", flush=True)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{args.reps} reps done", flush=True)

    if not results:
        args.out.write_text(json.dumps({
            "level": args.level, "n_ok": 0, "n_failed": failures,
            "error": "no usable measurement",
        }, indent=2), encoding="utf-8")
        sys.exit(1)

    totals = [r["total_mean"] for r in results]
    mean_total = round(statistics.fmean(totals), 3)
    subscales = {
        s: round(statistics.fmean(r["subscale_means"][s] for r in results), 3)
        for s in ISMI_SUBSCALES
    }
    row = {
        "level": args.level,
        "n_ok": len(results),
        "n_failed": failures,
        "ismi29_total_mean": mean_total,
        "ismi29_total_sd": round(statistics.stdev(totals), 3) if len(totals) > 1 else 0.0,
        "tier": interpret(mean_total),
        "subscale_means": subscales,
        "per_rep_totals": [round(t, 3) for t in totals],
        "per_rep_subscales": [r["subscale_means"] for r in results],
    }
    args.out.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] {args.level}: ISMI-29 = {mean_total} (SD {row['ismi29_total_sd']}) -> {row['tier']}")


if __name__ == "__main__":
    main()
