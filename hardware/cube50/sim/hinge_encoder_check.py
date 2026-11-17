"""Stray field at the hinge AS5600s (CAD positions, params.json "magnets" "encoders").

    nice -n 10 .venv/bin/python hardware/cube50/sim/hinge_encoder_check.py

1. Signal: the encoder's own 4 x 1 mm diametric magnet, first harmonic of Bz on the 1 mm Hall circle.
1. Own face magnets (48) and the other hinge's magnet, stowed and deployed.
1. Worst case: one neighbour face (8 magnets, in register orientation) slid to every in-plane
   position over the face the chip sits under (2 mm grid), chip stowed and deployed.
1. Pivot sweep: the cube turns 0..180 deg about each hinge edge with a neighbour in face contact on
   each side face (the faces that slide), neighbours at their lattice cells.
Error bound asin(stray / signal). Appends one row to magnet_results.jsonl.
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
import magnet_study as ms  # noqa: E402
import magnets as mg  # noqa: E402


def err(h, sig):
    return math.degrees(math.asin(min(1.0, h / sig)))


def main():
    lay = mg.layout()
    h = lay["half"]
    faces = mg.Magnets(*mg.cube_magnets(lay)[:2], lay["D"], lay["T"], lay["Br"])
    enc = mg.encoder_magnets(lay)
    names = [e["name"] for e in lay["encoders"]["magnets"]]
    off = np.array(lay["encoders"].get("deployed_offset_mm", [4.2, 0, 4.2])) / 1000
    row = {"check": "hinge_as5600", "skin_mm": lay["skin"] * 1e3}
    for chip, (die0, nrm, mag) in mg.chips(lay).items():
        k = names.index(mag)
        for state in ("stowed", "deployed"):
            shift = np.array([off[0], 0, off[2] * np.sign(die0[2])]) if state == "deployed" else np.zeros(3)
            die = die0 + shift
            own = enc.subset([k]).moved(np.eye(3), shift)
            sig = mg.as5600_error(lambda p: mg.field(own, p), die, nrm)[0]
            other = enc.subset([j for j in range(len(enc)) if j != k])
            h_own = mg.as5600_error(lambda p: mg.field(faces + other, p), die, nrm)[0]
            # neighbour face slid anywhere over the side face the chip sits under
            side = "-y" if nrm[1] > 0 and die0[1] < 0 else "+y"
            R, x = ms.place(side, "-x" if side == "+y" else "+x", 0, half=h)
            nb = mg.Magnets(*mg.face_magnets(lay, "-x" if side == "+y" else "+x"), lay["D"], lay["T"], lay["Br"]).moved(R, x)
            worst = 0.0
            for du in np.arange(-0.036, 0.0361, 0.002):
                for dv in np.arange(-0.036, 0.0361, 0.002):
                    src = nb.moved(np.eye(3), np.array([du, 0, dv]))
                    worst = max(worst, mg.as5600_error(lambda p: mg.field(src, p), die, nrm)[0])
            row[f"{chip}_{state}"] = {"die_mm": [round(v * 1e3, 2) for v in die], "signal_mt": round(sig * 1e3, 2), "own_faces_and_other_encoder_mt": round(h_own * 1e3, 4),
                                     "error_own_deg": round(err(h_own, sig), 3), "neighbour_anywhere_worst_mt": round(worst * 1e3, 3), "error_neighbour_anywhere_deg": round(err(worst, sig), 2)}
    # pivot sweep: the cube turns about its top hinge edge (x = h, z = h, along y); neighbours on +-y stay
    nbs = None
    for s in (1, -1):
        R, x = ms.place("+y" if s > 0 else "-y", "-y" if s > 0 else "+y", 0, half=h)
        n = mg.cube_set(lay).moved(R, x)
        nbs = n if nbs is None else nbs + n
    sweep = {}
    for edge_z in (h, -h):
        p = np.array([h, 0, edge_z])
        worst = {c: 0.0 for c in mg.chips(lay)}
        for a in np.arange(0, 180.1, 2.0):
            Rr = Rotation.from_rotvec(-math.radians(a) * np.sign(edge_z) * np.array([0, 1.0, 0])).as_matrix()
            for chip, (die0, nrm, mag) in mg.chips(lay).items():
                deployed = (die0[2] > 0) == (edge_z > 0)  # the pivoting hinge is deployed
                shift = np.array([off[0], 0, off[2] * np.sign(die0[2])]) if deployed else np.zeros(3)
                die = Rr @ (die0 + shift - p) + p
                worst[chip] = max(worst[chip], mg.as5600_error(lambda q: mg.field(nbs, q), die, Rr @ nrm)[0])
        sweep["top_hinge" if edge_z > 0 else "bottom_hinge"] = {c: round(v * 1e3, 4) for c, v in worst.items()}
    row["pivot_sweep_side_neighbours_mt"] = sweep
    ms.emit(row)


if __name__ == "__main__":
    main()
