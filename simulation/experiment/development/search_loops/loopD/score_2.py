"""Round-2 autoresearch score for one finished evaluation run (results/v3_runs/<run>/_score.json + the notes).

SCORE = 1 - E exactly as in round 1 (score.py). GUARD = round 1's per-cell checks AND a content check on the
notes, because E only sees the judged number and is blind to what the note says:

  copy1   new note sentences that copy another speaker's FIRST-person sentence near-verbatim
          (appropriation candidates: someone else's experience written as the patient's own)      <= 3 per run
  copy2   new note sentences that paste another speaker's SECOND-person remark near-verbatim       <= 20 per run
  words   longest final (session-8) note                                                          <= 250 words
  sents   final notes keep 4..9 sentences

Thresholds were set 10-15 from existing runs of the same eval (copy1: paper code 6 / 0 / 1, loop-3 final 2 / 0;
copy2: paper 1 / 1 / 6, loop-3 final 17 / 9, a rejected loop-3 change 46 / 38; words: paper max 179-202,
a rejected change 499; sentences: paper 4-8). Copy matching is the loop-3 copy_scan rule
(difflib ratio >= 0.75 against the other speakers' sentences of that session, 5+ word sentences).
Paraphrased appropriation is not caught here -- kept versions are also read by a person afterwards.
Prints `SCORE: X.XXXX` on its own line.
"""
import glob
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

B = Path(__file__).resolve().parent.parent
LIMITS = {"copy1": 3, "copy2": 20, "words": 250, "sents": (4, 9)}

_split = lambda t: [x.strip() for x in re.split(r'(?<=[.!?])\s+|\s*\|\s*', t) if len(x.strip().split()) >= 5]
_norm = lambda t: re.sub(r"[^a-z' ]", ' ', t.lower().replace('’', "'")).split()
FIRST = re.compile(r"\b(i|i'm|i've|i'd|my|me|myself)\b")
SECOND = re.compile(r"\b(you|you're|you'd|your|yourself)\b")


def _best(sent, others):
    a = ' '.join(_norm(sent))
    top = (0.0, '')
    for o in others:
        b = ' '.join(_norm(o))
        r = SequenceMatcher(None, a, b).ratio()
        if b and b in a:
            r = max(r, 0.9)
        top = max(top, (r, o))
    return top


def content(run_dir: Path) -> dict:
    dlg = {}
    for line in open(B / 'testset_v3' / 'dev.jsonl'):
        if line.strip():
            d = json.loads(line)
            dlg[d['test_id']] = {s['session']: s['turns'] for s in d['visible_dialogue']}
    c1 = c2 = 0
    ex1 = []
    words, sents = [], []
    for f in sorted(glob.glob(str(run_dir / 'C*.json'))):
        tid = os.path.basename(f)[:-5]
        for ri, r in enumerate(json.load(open(f))['runs']):
            ss = r['sessions']
            last = ss[-1]['belief_narrative']
            words.append(len(last.split()))
            sents.append(len([x for x in re.split(r'(?<=[.!?])\s+', last.strip()) if x.strip()]))
            for i in range(1, len(ss)):
                if not ss[i].get('changed'):
                    continue
                old = {' '.join(_norm(x)) for x in _split(ss[i - 1]['belief_narrative'])}
                others = [x for t in dlg[tid].get(i, []) if t['speaker'] != 'patient' for x in _split(t['text'])]
                for s in _split(ss[i]['belief_narrative']):
                    if ' '.join(_norm(s)) in old:
                        continue
                    sc, o = _best(s, others)
                    if sc >= 0.75:
                        w = ' '.join(_norm(o))
                        if FIRST.search(w):
                            c1 += 1
                            ex1.append(f'{tid} r{ri} S{i}: {s[:120]}')
                        elif SECOND.search(w):
                            c2 += 1
    return {"copy1": c1, "copy2": c2, "words": max(words), "sents": (min(sents), max(sents)), "copy1_examples": ex1}


def main() -> None:
    run = sys.argv[1]
    rd = B / 'results' / 'v3_runs' / run
    p = rd / '_score.json'
    if not p.exists():
        print(f"CRASH: no score file for {run} (see results/v3_runs/_{run}.log)")
        sys.exit(1)
    s = json.loads(p.read_text())
    if s.get("missing"):
        print(f"CRASH: dialogues missing {s['missing']}")
        sys.exit(1)
    k = content(rd)
    ok = {
        "copy1": k["copy1"] <= LIMITS["copy1"],
        "copy2": k["copy2"] <= LIMITS["copy2"],
        "words": k["words"] <= LIMITS["words"],
        "sents": LIMITS["sents"][0] <= k["sents"][0] and k["sents"][1] <= LIMITS["sents"][1],
    }
    cells_ok = bool(s["guards_pass"])
    print(f"E: {s['E']:.4f}")
    print("GUARD: " + ("PASS" if cells_ok and all(ok.values()) else "FAIL"))
    print("  cells: " + ("PASS" if cells_ok else "FAIL"))
    for c in s["cells"]:
        print(f"    cell {c['cell']} {c['kind']:18s} start {c['start']}  mean change {c['mean_delta']:+.3f}  {'ok' if c['ok'] else 'FAIL'}")
    print("  content: " + ("PASS" if all(ok.values()) else "FAIL"))
    print(f"    copy1 first-person copies of others {k['copy1']} (<= {LIMITS['copy1']})  {'ok' if ok['copy1'] else 'FAIL'}")
    for e in k["copy1_examples"][:5]:
        print(f"      {e}")
    print(f"    copy2 second-person pastes {k['copy2']} (<= {LIMITS['copy2']})  {'ok' if ok['copy2'] else 'FAIL'}")
    print(f"    words longest final note {k['words']} (<= {LIMITS['words']})  {'ok' if ok['words'] else 'FAIL'}")
    print(f"    sents final notes {k['sents'][0]}-{k['sents'][1]} (4..9)  {'ok' if ok['sents'] else 'FAIL'}")
    for kk, v in s["kinds"].items():
        print(f"  kind {kk:18s} mean change {v['mean_delta']:+.3f}  share of sessions changed {v['changed_share']:.2f}")
    print(f"SCORE: {1 - s['E']:.4f}")


if __name__ == "__main__":
    main()
