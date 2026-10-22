"""Replay the released run records into a Minibook instance, one post per session, so the reported threads can be
browsed as platform threads.

The reported runs generated each session's thread in process with Minibook's structure (one opening post by the
Patient, then replies); this script writes those exact texts to Minibook through its REST API and reads every thread
back to check it. It replays; it does not generate anything.

  python3 setup_minibook.py --base-url http://localhost:3456/api/v1 --out minibook.local.json
  python3 replay_to_minibook.py --config minibook.local.json --runs ../results/runs --levels 270 --conditions peer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agents"))          # io_utils, used by minibook_client
from minibook_client import MinibookThreadStore          # noqa: E402


def with_retry(call, *args):
    """Minibook rate-limits each agent (by default 10 posts and 60 comments per minute); wait and retry on 429."""
    for _ in range(30):
        try:
            return call(*args)
        except RuntimeError as exc:
            if "(429)" not in str(exc):
                raise
            time.sleep(10)
    raise RuntimeError("Minibook kept refusing requests (429); raise rate_limits in its config.yaml")


def replay(store: MinibookThreadStore, turns: list[dict]) -> int:
    sessions = sorted({int(t["session"]) for t in turns})
    for s in sessions:
        thread = [t for t in turns if int(t["session"]) == s]
        if not thread or thread[0]["speaker"] != "patient":
            raise RuntimeError(f"session {s} does not open with a Patient post")
        with_retry(store.start_session, s, thread[0]["content"])
        for t in thread[1:]:
            with_retry(store.add_comment, s, t["speaker"], t["content"])
        back = store.read_session(s)
        if [(b["speaker"], b["content"]) for b in back] != [(t["speaker"], t["content"]) for t in thread]:
            raise RuntimeError(f"session {s}: the thread read back from Minibook differs from the run record")
    return len(sessions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="written by setup_minibook.py")
    ap.add_argument("--runs", type=Path, default=HERE.parent / "results" / "runs")
    ap.add_argument("--conditions", nargs="*", default=["peer", "general"])
    ap.add_argument("--levels", nargs="*", type=int, help="e.g. 270 (Noor, severe); default: all")
    ap.add_argument("--out", type=Path, default=HERE / "replay_manifest.json")
    a = ap.parse_args()
    manifest = []
    for cond in a.conditions:
        for f in sorted((a.runs / cond).glob("level_*/run_[0-9][0-9].json")):
            level = int(f.parent.name.split("_")[1])
            if a.levels and level not in a.levels:
                continue
            rec = json.loads(f.read_text())
            tid = f"{cond}-level{level}-{f.stem}"
            store = MinibookThreadStore.from_file(a.config, tid)
            n = replay(store, rec["turns"])
            manifest.append(store.public_metadata())
            print(f"{tid}: {n} sessions replayed and verified")
    a.out.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
