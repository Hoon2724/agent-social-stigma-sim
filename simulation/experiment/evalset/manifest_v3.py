"""Write testset_v3/MANIFEST.json after assembly (check_v3.py summary --assemble):
per dialogue -- cell, type, start/target, person, attempt, sha256 of the visible dialogue,
session-0 note key/sha/pooled judge mean, and the acceptance-check numbers. Test-split rows
carry the same fields (hashes and numbers only, no text), so the sealed file can be verified
later without opening it now.

  python3 manifest_v3.py
"""

from __future__ import annotations

import hashlib
import json
import statistics

import check_v3
import harness
import v3data


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def rows(split: str) -> list[dict]:
    path = (v3data.SEALED / "test.jsonl") if split == "test" else (v3data.DEV_DIR / "dev.jsonl")
    if not path.exists():
        return []
    d = check_v3.check_dir(split)
    direc, judged, review = (check_v3.load_json(d / f, {}) for f in ("direction.json", "oracle_judged.json", "review.json"))
    pooled = check_v3.load_json((v3data.SEALED if split == "test" else v3data.DEV_DIR) / "s0_pooled.json", {})
    exc = check_v3.load_json((v3data.SEALED if split == "test" else v3data.DEV_DIR) / "EXCEPTIONS.json", {})
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        k, c = check_v3.key(it), it["condition"]
        s0 = check_v3.s0_text(it)
        s0_sha = harness.text_hash(s0)
        deltas = [v["delta"] for kk, v in judged.items() if kk.split("/")[0] == k and v.get("delta") is not None]
        out.append({
            "test_id": it["test_id"], "split": split, "cell": it["cell"], "kind": c["trajectory_type"],
            "start": c["initial_ismi"], "target_final": c["target_final_ismi"], "persona_id": c["persona_id"],
            "attempt": it["attempt"], "attempts_made": sum(1 for x in check_v3.attempts(split, [it["test_id"]])),
            "generator": it["model"], "prompt_version": it["prompt_version"],
            "visible_dialogue_sha256": sha(it["visible_dialogue"]),
            "s0_key": c["s0_key"], "s0_text_sha": s0_sha, "s0_pooled_mean": pooled.get(s0_sha, {}).get("mean"),
            "s0_pooled_n": pooled.get(s0_sha, {}).get("n"),
            "direction_ratings": direc.get(k, {}).get("ratings"),
            "oracle_deltas": deltas, "oracle_delta_mean": round(statistics.fmean(deltas), 3) if deltas else None,
            "review": {x: review.get(k, {}).get(x) for x in ("pass", "problems", "severity")},
            "accepted_by": "user_exception" if it["test_id"] in exc else "checks",
            "exception": exc.get(it["test_id"]),
        })
    return out


def main() -> None:
    m = {"dataset": "belief_v3", "plan": "docs/superpowers/plans/2026-10-13-belief-eval-dataset.md",
         "note": "test split is sealed (not in the repository; neither person opens it before the final "
                 "comparison) -- only hashes and check numbers are listed here",
         "dev": rows("dev"), "test": rows("test")}
    (v3data.DEV_DIR / "MANIFEST.json").write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")
    print(f"MANIFEST: dev {len(m['dev'])}, test {len(m['test'])}")


if __name__ == "__main__":
    main()
