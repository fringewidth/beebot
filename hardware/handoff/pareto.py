"""Per setting: ten-cube harm/reached (all, and without lone-support moves), climb reach, floor-roll reach,
mean attempts; store-weighted harm/reached. Marks ten-cube points dominated by another."""
import json, sys, glob, os
from collections import Counter, defaultdict
from pathlib import Path
from beebot.hardware.cube_outcomes import lattice_end, setting_of, weighted, GROUPS
d = sys.argv[1]
weights = json.loads((Path(d) / "../draw/weights.json").read_text())["counts"]
by = defaultdict(lambda: {"layouts": [], "stores": []})
for p in sorted(glob.glob(f"{d}/*.jsonl")):
    if "right-" in p: continue
    for line in open(p):
        r = json.loads(line)
        if r.get("movers", 1) != 1 or r["detail"] == "refused" or r["magnets"] != "release": continue
        if r["set"] in ("layouts", "stores"):
            name = setting_of(r)
            if os.environ.get("POOL"): name = name.split("@")[0]
            by[name][r["set"]].append(r)
out = {}
for s, sets in by.items():
    w = sets["layouts"]
    row = {}
    if w:
        e = [lattice_end(r)[0] for r in w]
        harm = lambda idx: sum(e[i] not in ("reached", "stayed") for i in idx) / max(len(idx), 1)
        reach = lambda idx: sum(e[i] == "reached" for i in idx) / max(len(idx), 1)
        allx = range(len(w)); nl = [i for i in allx if "lone" not in w[i]["key"]]
        g = lambda name: [i for i in allx if GROUPS[name](w[i]["key"])]
        row.update(n=len(w), harm=harm(allx), reached=reach(allx), harm_nolone=harm(nl), reached_nolone=reach(nl),
                   cvx_up=reach(g("convex up")), ccv_up=reach(g("concave up")), cvx_down=reach(g("convex down")), free_roll=reach(g("floor roll, free")),
                   attempts=sum(r.get("attempts", 1) for r in w) / len(w))
    if sets["stores"]:
        m, _, n = weighted(sets["stores"], weights, GROUPS)["all"]
        row.update(s_harm=1 - m[0] - m[1], s_reached=m[0])
    out[s] = row
pts = {s: r for s, r in out.items() if "harm" in r}
for s, r in sorted(out.items(), key=lambda kv: kv[1].get("harm", 9)):
    dom = any(o["harm"] <= r["harm"] and o["reached"] >= r["reached"] and (o["harm"] < r["harm"] or o["reached"] > r["reached"]) for t, o in pts.items() if t != s and t.split("@")[0] != s.split("@")[0] and ("@s1" in t) == ("@s1" in s)) if "harm" in r else False
    f = lambda k: f"{r[k]:.3f}" if k in r else "  -  "
    print(f"{s:22s} {'dom' if dom else '   '} 10c harm {f('harm')} reach {f('reached')} | nolone {f('harm_nolone')}/{f('reached_nolone')} | cvx_up {f('cvx_up')} ccv_up {f('ccv_up')} cvx_dn {f('cvx_down')} free {f('free_roll')} att {f('attempts')} | stores {f('s_harm')}/{f('s_reached')}")
