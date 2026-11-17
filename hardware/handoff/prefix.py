"""Per primitive group, reached / harmful of escalated rows cut to each prefix of the schedule."""
import json, sys
from collections import Counter
from beebot.hardware.cube_outcomes import GROUPS, plan_outcomes
for p in sys.argv[1:]:
    rows = [json.loads(l) for l in open(p)]
    print("==", p)
    for label, member in GROUPS.items():
        cells = {}
        for r in rows:
            if not member(r["key"]): continue
            for plan, end in plan_outcomes(r):
                c = cells.setdefault(plan, Counter()); c["n"] += 1; c["r"] += end == "reached"; c["h"] += end not in ("reached", "stayed")
        print(f"{label:20s}" + "  ".join(f"{len(k)}:{c['r']/c['n']:.2f}/{c['h']/c['n']:.2f}" for k, c in sorted(cells.items(), key=lambda kv: len(kv[0]))) + f"  (n {max(c['n'] for c in cells.values()) if cells else 0})")
