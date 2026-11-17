"""Harm per pivot in ten-cube walk rows, split: knocks where a knocked cube only dropped (fell to a
lower level: a cube held up by the mover alone), and the rest. Usage: python drop.py rows.jsonl ..."""
import json
import sys
from collections import Counter

for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r["detail"] != "refused"]
    n = len(rows)
    c = Counter()
    for r in rows:
        if r["detail"] in ("reached", "stayed"):
            c[r["detail"]] += 1
        elif r["detail"] == "knock" and all(k["end"][0] < k["start"][0] for k in r["knocked"]):
            c["dropped"] += 1
        else:
            c[r["detail"]] += 1
    harm = n - c["reached"] - c["stayed"]
    print(f"{path.split('/')[-1]:32s} n {n:4d} reached {c['reached'] / n:.3f} harm {harm / n:.3f}  of which dropped {c['dropped'] / n:.3f}, knock {c['knock'] / n:.3f}, off {c['off'] / n:.3f}, strayed {c['strayed'] / n:.3f}; harm without drops {(harm - c['dropped']) / n:.3f}")
