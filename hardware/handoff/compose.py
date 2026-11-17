"""Compose store rows run at several kicks into the rows a kick table would give (in-sample)."""
import json, sys
from beebot.hardware.cube_outcomes import factor_of
table = json.load(open(sys.argv[1])); name = sys.argv[2]; out = open(sys.argv[3], "w")
seen = set()
for p in sys.argv[4:]:
    for line in open(p):
        r = json.loads(line)
        key = r["key"]; prim = key.split("/")[0]
        want = table.get(key, table.get(prim, 1.0))
        tag = (r.get("store"), r.get("episode"), r.get("t"), r["seed"], tuple(r["start"]))
        if abs(factor_of(r) - want) < 1e-6 and tag not in seen:
            seen.add(tag); r["setting"] = name; out.write(json.dumps(r) + "\n")
print(len(seen), "rows")
