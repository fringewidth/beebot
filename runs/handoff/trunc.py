"""Escalated rows cut to their first N attempts (a move still staying after N ends as stayed); prints mean attempts."""
import json, sys
n = int(sys.argv[1]); out = open(sys.argv[2], "w"); att = []
for p in sys.argv[3:]:
    for line in open(p):
        r = json.loads(line)
        if r.get("attempts", 1) > n:
            r = {**r, "after": r["cells"], "off_after": r.get("off_before", []), "end": r["start"], "mover": "stayed", "detail": "stayed", "knocked": [], "attempts": n, "tries": r["tries"][:n]}
        r["setting"] = r["setting"] + f"@{n}"
        att.append(r.get("attempts", 1)); out.write(json.dumps(r) + "\n")
print(f"mean attempts {sum(att)/len(att):.2f} over {len(att)} moves")
