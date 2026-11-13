"""Statistics of the machine-parts library for the physics brainstorm (JSON on stdout, one line to the log).

    .venv/bin/python scripts/machine_parts_stats.py [--log]

Reads data/machine_parts/{fusion360,mcb}_parts.jsonl, fusion360_assembly_table.jsonl and the accepted-variant
files. Name rule (stated): a name is auto-generated when it is empty, a bare number or hex id, or a CAD default
(Body12, Component3:1, Part1, Untitled ...; machine_parts.name_is_generic); every other name counts as human.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import machine_parts as MP

ROOT = Path("data/machine_parts")
LOG = Path("logs/machine-parts.out")


def rows(name):
    path = ROOT / name
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path)]


def quant(values, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    if len(values) == 0:
        return []
    return [round(float(q), 3) for q in np.quantile(np.asarray(values, dtype=float), qs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    out = {}
    fusion = rows("fusion360_parts.jsonl")
    asm = rows("fusion360_assembly_table.jsonl")
    kinds_joint = Counter()
    kinds_asm = Counter()
    for a in asm:
        kinds_joint.update(a["joint_kinds"])
        kinds_asm.update(a["joint_kinds"].keys())
    out["fusion360"] = {
        "assemblies": len(asm),
        "assemblies_with_joints": sum(1 for a in asm if a["joints"]),
        "joints": sum(a["joints"] for a in asm),
        "joint_histogram": dict(kinds_joint.most_common()),
        "joint_histogram_share": {k: round(v / max(sum(kinds_joint.values()), 1), 4) for k, v in kinds_joint.most_common()},
        "assemblies_containing_kind": dict(kinds_asm.most_common()),
        "assemblies_with_a_free_joint": sum(1 for a in asm if set(a["joint_kinds"]) & MP.FREE_KINDS),
        "bodies_per_assembly_quantiles": quant([a["bodies"] for a in asm]),
        "holes_per_assembly_quantiles": quant([a["holes"] for a in asm]),
    }
    free = Counter()
    for r in fusion:
        kinds = {j["kind"] for j in r["joints"]}
        if not kinds:
            free["no_joint"] += 1
        elif kinds & MP.FREE_KINDS:
            free["takes_part_in_a_free_joint"] += 1
        else:
            free["only_rigid_joints"] += 1
    total = max(len(fusion), 1)
    out["fusion360"]["parts"] = len(fusion)
    out["fusion360"]["part_mobility"] = dict(free)
    out["fusion360"]["part_mobility_share"] = {k: round(v / total, 4) for k, v in free.items()}
    jointed = [r for r in fusion if r["joints"]]
    out["fusion360"]["jointed_parts_free_share"] = round(sum(1 for r in jointed if {j["kind"] for j in r["joints"]} & MP.FREE_KINDS) / max(len(jointed), 1), 4)
    # names
    quality = {}
    for field in ("name", "occurrence_name", "component_name"):
        vals = [r[field].split(":")[0] for r in fusion]
        quality[field + "_human_share"] = round(sum(1 for v in vals if not MP.name_is_generic(v)) / total, 4)
    quality["any_human_name_share"] = round(sum(1 for r in fusion if r["label"]) / total, 4)
    humans = [r["label"] for r in fusion if r["label"]]
    gens = [r["name"] or r["occurrence_name"] for r in fusion if not r["label"]]
    rng = np.random.default_rng(0)
    quality["human_examples"] = [humans[i] for i in rng.choice(len(humans), min(25, len(humans)), replace=False)] if humans else []
    quality["generic_examples"] = [gens[i] for i in rng.choice(len(gens), min(12, len(gens)), replace=False)] if gens else []
    quality["most_common_human_names"] = Counter(humans).most_common(25)
    out["fusion360"]["names"] = quality
    out["fusion360"]["materials"] = Counter(r["material"] or "(none)" for r in fusion).most_common(12)
    out["fusion360"]["parent_depth_hist"] = dict(sorted(Counter(r["depth"] for r in fusion).items()))
    meshed = [r for r in fusion if r["dims_mm"]]
    out["fusion360"]["parts_with_mesh"] = len(meshed)
    out["fusion360"]["dims_mm_longest_quantiles"] = quant([max(r["dims_mm"]) for r in meshed])
    out["fusion360"]["dims_mm_shortest_quantiles"] = quant([min(r["dims_mm"]) for r in meshed])
    out["fusion360"]["shape_class"] = dict(Counter(r.get("shape_class") for r in meshed).most_common())
    out["fusion360"]["aspect_longest_over_shortest_quantiles"] = quant([max(r["dims_mm"]) / max(min(r["dims_mm"]), 1e-6) for r in meshed])
    # mcb
    mcb = rows("mcb_parts.jsonl")
    out["mcb"] = {
        "parts": len(mcb),
        "classes": len({r["label"] for r in mcb}),
        "class_histogram": dict(Counter(r["label"] for r in mcb).most_common()),
        "shape_class": dict(Counter(r.get("shape_class") for r in mcb).most_common()),
        "aspect_longest_over_shortest_quantiles": quant([max(r["dims_norm"]) / max(min(r["dims_norm"]), 1e-6) for r in mcb if r.get("dims_norm")]),
    }
    # acceptance
    for source, parts in (("fusion360", fusion), ("mcb", mcb)):
        done = [r for r in parts if r.get("accepted") is not None]
        zero = [r for r in done if r["accepted"] == 0]
        why = Counter()
        for r in zero:
            reject = r.get("reject") or {}
            why[max(reject, key=reject.get) if reject and max(reject.values()) else "none"] += 1
        skipped = sum(1 for r in parts if r.get("skipped"))
        lines = G.load(source, root=ROOT)[0] if (ROOT / f"{source}.jsonl").exists() else []
        thick = [l["extra"]["thickness_cells"] for l in lines]
        cells = [l["cells"] for l in lines]
        rec = {
            "parts_voxelised": len(done),
            "parts_with_accepted_variant": len(done) - len(zero),
            "accepted_share_of_voxelised": round((len(done) - len(zero)) / max(len(done), 1), 4),
            "parts_without_variant": len(zero),
            "main_reject_reason_of_those": dict(why),
            "shape_class_of_those": dict(Counter(r.get("shape_class") for r in zero)),
            "skipped_too_many_faces": skipped,
            "variants_accepted": len(lines),
            "cells_hist": G.histogram(cells, edges=(4, 16, 32, 64, 100, 150, 250, 400, 600, 801)),
            "thickness_cells_quantiles": quant(thick),
            "variants_thinner_than_1_cell": round(float(np.mean(np.array(thick) < 1.0)), 4) if thick else None,
            "variants_thinner_than_2_cells": round(float(np.mean(np.array(thick) < 2.0)), 4) if thick else None,
            "variants_one_cell_thick_bbox": round(float(np.mean([min(l["dims"]) == 1 for l in lines])), 4) if lines else None,
            "variants_two_cells_or_less_bbox": round(float(np.mean([min(l["dims"]) <= 2 for l in lines])), 4) if lines else None,
            "cells_of_variants_quantiles": quant(cells),
        }
        by_shape = {}
        for shape in ("cube", "block", "rod", "plate", "strip"):
            sel = [l["extra"]["thickness_cells"] for l in lines if l["extra"].get("shape_class") == shape]
            if sel:
                by_shape[shape] = {"n": len(sel), "thickness_cells_median": quant(sel, (0.5,))[0], "share_below_1_cell": round(float(np.mean(np.array(sel) < 1.0)), 3)}
        rec["by_shape_class"] = by_shape
        out[source]["acceptance"] = rec
    out["machine_parts_accepted"] = sum(out[s]["acceptance"]["variants_accepted"] for s in ("fusion360", "mcb"))
    print(json.dumps(out, indent=1, default=str))
    if args.log:
        with open(LOG, "a") as handle:
            handle.write(json.dumps({"stage": "final", "machine_parts_accepted": out["machine_parts_accepted"], "fusion360_joint_histogram": out["fusion360"]["joint_histogram"], "fusion360_part_mobility_share": out["fusion360"]["part_mobility_share"], "names_any_human_share": quality["any_human_name_share"], "mcb_parts": out["mcb"]["parts"], "mcb_classes": out["mcb"]["classes"]}) + "\n")


if __name__ == "__main__":
    main()
