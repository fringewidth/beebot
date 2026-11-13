"""Recompute the Fusion 360 part labels with the current name rule (machine_parts.name_is_generic) in the
stored files, without redoing the conversion: fusion360_parts.jsonl, fusion360.jsonl (variants) and
fusion360_assemblies.jsonl. The label is the first human-chosen name among occurrence, component and body name
(the ":1" instance suffix removed); empty when all three are auto-generated.

    .venv/bin/python scripts/machine_parts_relabel.py
"""

import json
from pathlib import Path

from beebot.vla.voxel import machine_parts as MP

ROOT = Path("data/machine_parts")


def label_of(row):
    for field in ("occurrence_name", "component_name", "name"):
        text = row[field].split(":")[0].strip()
        if text and not MP.name_is_generic(text):
            return text
    return ""


def main():
    rows = [json.loads(line) for line in open(ROOT / "fusion360_parts.jsonl")]
    labels = {}
    changed = 0
    for row in rows:
        new = label_of(row)
        changed += new != row["label"]
        row["label"] = new
        labels[row["part_id"]] = new
    with open(ROOT / "fusion360_parts.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    lines = [json.loads(line) for line in open(ROOT / "fusion360.jsonl")]
    for line in lines:
        line["label"] = labels.get(line["source_id"], "")
        line["text"] = line["label"]
        line["item"] = f"{line['source']}/{line['source_id']}/{line['kind']}/{line['label']}"
    with open(ROOT / "fusion360.jsonl", "w") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
    path = ROOT / "fusion360_assemblies.jsonl"
    asm = [json.loads(line) for line in open(path)]
    by_asm = {}
    for row in rows:
        by_asm.setdefault(row["assembly"], {})[row["index"]] = row
    for entry in asm:
        for part in entry["parts"]:
            row = by_asm[entry["assembly"]][part["index"]]
            part["name"] = row["label"] or row["name"]
            part["generic"] = not row["label"]
    with open(path, "w") as handle:
        for entry in asm:
            handle.write(json.dumps(entry) + "\n")
    print(json.dumps({"parts": len(rows), "labels_changed": changed, "with_label": sum(1 for r in rows if r["label"])}))


if __name__ == "__main__":
    main()
