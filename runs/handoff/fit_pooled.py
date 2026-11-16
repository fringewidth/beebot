"""Fit an outcome table on rows of several setting names (seeds) at once."""
import json, sys
from pathlib import Path
from beebot.hardware import cube_outcomes as C
name, out, *paths = sys.argv[1:]
rows = [json.loads(l) for p in paths for l in open(p)]
rows = [r for r in rows if r["magnets"] == "release" and r["set"] in ("stores", "layouts")]
table = C.fit(rows, name, "release")
table["sources"] = paths
table["setting"] = "m14-sel"
for e in table["classes"].values():
    e["lattice"], e["detail"] = dict(e["lattice"]), dict(e["detail"])
Path(out).write_text(json.dumps(table, separators=(",", ":")) + "\n")
print(table["rows"], "rows", len(table["classes"]), "classes")
