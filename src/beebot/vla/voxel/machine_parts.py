"""Machine parts: constituents of engineering assemblies, voxelised to the swarm's 16^3 grid.

Sources (see runs/handoff/machine-parts.md): the Fusion 360 Gallery Assembly dataset (designer-named bodies, an
assembly tree, joints with types) and the Mechanical Components Benchmark (MCB; parts in expert-labelled
classes). A part is a triangle mesh plus what the source states about it. Each part is oriented (resting on
its largest flat face, or in the file's own frame with z up), voxelised on a fine grid, and handed to
goal_library.variants, which re-voxelises at several scales and keeps those that fit 16 per axis, hold 4-800
cells and are face-connected to the floor. Storage is goal_library's (`save`, `load`), per source.

Units: meshes are taken in the source's unit; `unit_mm` converts to millimetres (Fusion 360 files are cm).
Nothing here invents a function for a part: `shape_class` is measured from the oriented bounding box only.
"""

import re
from collections import Counter

import numpy as np

from beebot.vla.voxel import goal_library

PART_MAX_CELLS = 800  # parts may be bigger than goals: recognisable shapes need cells
PART_TARGETS = (100, 250, 600)  # per part, the accepted scale nearest each of these cell counts is kept
PART_SCALES = tuple(sc for sc in goal_library.SCALES if sc <= 1.0)  # the fine grid is the finest; never upsample
FINE = 64  # cells along the longest axis of the fine grid every part is first voxelised to
FLAT_SHARE = 0.05  # a hull facet group counts as a flat face to rest on when it has this share of the hull area
GENERIC_NAME = re.compile(
    r"^\s*(body|component|part|solid|surface|bodies|untitled|unnamed|occurrence|assembly|root|prt|asm|"
    r"copy of body|extrusion|sketch|mesh|object|group|node|shape|brep)[\s_\-:.#(]*\d*\)?(\s*v\d+)?(\s*:\d+)?\s*$",
    re.IGNORECASE,
)
HEXISH = re.compile(r"^[0-9a-f\-]{8,}$", re.IGNORECASE)


def name_is_generic(name):
    """True for auto-generated names: empty, "Body12", "Component3:1", "Untitled", a bare number or a hex id.
    Anything else is counted as a name a person chose (it may still be uninformative: this is only a rule)."""
    text = (name or "").strip()
    if not text:
        return True
    if GENERIC_NAME.match(text):
        return True
    if text.isdigit():
        return True
    if HEXISH.match(text):
        return True
    return False


def shape_class(dims):
    """Measured role from the sorted bounding box (a <= b <= c): rod (c much longer than a and b), plate (a much
    thinner than b and c), block (neither), cube-like when all three are within 1.5x. Thresholds: 0.35 and 2.5."""
    a, b, c = sorted(float(d) for d in dims)
    if c <= 0:
        return "empty"
    if c / max(a, 1e-9) <= 1.5:
        return "cube"
    thin = a / max(b, 1e-9) < 0.35
    long = c / max(b, 1e-9) > 2.5
    if thin and long:
        return "strip"
    if thin:
        return "plate"
    if long:
        return "rod"
    return "block"


# ---------------------------------------------------------------------------------------------
# Orientation and voxelising


def rotation_aligning(vec, target):
    """Rotation matrix taking unit vector `vec` to unit vector `target`."""
    vec = np.asarray(vec, dtype=float)
    target = np.asarray(target, dtype=float)
    vec = vec / np.linalg.norm(vec)
    target = target / np.linalg.norm(target)
    cross = np.cross(vec, target)
    dot = float(vec @ target)
    if np.linalg.norm(cross) < 1e-9:
        if dot > 0:
            return np.eye(3)
        axis = np.eye(3)[int(np.argmin(np.abs(vec)))]
        axis = np.cross(vec, axis)
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))


def flat_face_normal(mesh, share=FLAT_SHARE):
    """Outward normal of the largest planar face of the convex hull (coplanar hull triangles grouped), or None
    when no group holds `share` of the hull area."""
    hull = mesh.convex_hull
    facets = hull.facets
    if len(facets) == 0:
        return None
    areas = hull.facets_area
    best = int(np.argmax(areas))
    if areas[best] < share * hull.area:
        return None
    return hull.facets_normal[best]


def orient_flat(vertices, faces):
    """Vertices turned so the largest hull face (else the shortest bounding-box axis) points down, in the (x, y, z)
    frame with z up. Returns (vertices, how) with how in {"face", "short_axis"}."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normal = flat_face_normal(mesh)
    how = "face"
    if normal is None:
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        normal = np.eye(3)[int(np.argmin(extent))]
        how = "short_axis"
    rot = rotation_aligning(normal, [0.0, 0.0, -1.0])
    return vertices @ rot.T, how


def voxelise_mesh(vertices, faces, fine=FINE, pitch=None):
    """Boolean occupancy (z, y, x) of a mesh: surface voxels of a subdivided mesh at `pitch` (default: longest
    extent / fine), interior filled. Cells are a cube lattice; returns (vox, pitch)."""
    import trimesh
    from scipy import ndimage
    from trimesh.voxel import creation

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    extent = mesh.bounds[1] - mesh.bounds[0]
    if pitch is None:
        pitch = float(extent.max()) / fine
    if pitch <= 0:
        return np.zeros((1, 1, 1), dtype=bool), 1.0
    grid = creation.voxelize_subdivide(mesh, pitch)
    dense = np.asarray(grid.matrix, dtype=bool)
    filled = ndimage.binary_fill_holes(np.pad(dense, 1))[1:-1, 1:-1, 1:-1]
    return np.ascontiguousarray(filled.transpose(2, 1, 0)), pitch


def scaled_variants(vox, rng, keep=PART_TARGETS, rotations=2, max_cells=PART_MAX_CELLS, scales=PART_SCALES):
    """goal_library.variants restricted to `scales` (parts are already on a fine grid: never upsample, which
    also keeps memory down), using goal_library.best_variant per scale. Same output: (scale, rot, vox) list and
    a reject tally."""
    base = goal_library.crop_bbox(vox)
    tally = {"oversize": 0, "small": 0, "big": 0, "ungrounded": 0}
    if base is None:
        return [], tally
    seen = {}
    for scale in scales:
        found = goal_library.best_variant(base, scale, max_cells)
        if isinstance(found, str):
            tally[found] += 1
            continue
        scaled = found["vox"]
        seen.setdefault((scaled.shape, scaled.tobytes()), (scale, scaled))
    pool = sorted(seen.values(), key=lambda it: int(it[1].sum()))
    chosen = {}
    for target in keep:
        if not pool:
            break
        best = min(pool, key=lambda it: abs(int(it[1].sum()) - target))
        chosen.setdefault(round(best[0], 4), best)
    out = []
    for scale, scaled in chosen.values():
        out.append((scale, 0, np.ascontiguousarray(scaled)))
        if rotations > 1:
            k = int(rng.integers(1, 4))
            turned = np.ascontiguousarray(goal_library.rotate(scaled, k))
            if not (turned.shape == scaled.shape and (turned == scaled).all()):
                out.append((scale, k, turned))
    return out, tally


def part_variants(vertices, faces, dims_mm, rng, fine=FINE):
    """Accepted goal_library variants of one part in two orientations, as dicts with vox, scale, rot,
    orientation, cell_mm, thickness_cells (the thinnest bounding extent measured in cells) and the reject
    tally. `dims_mm` is the bounding box of the mesh in mm (any orientation)."""
    out = []
    tally = Counter()
    flat, how = orient_flat(vertices, faces)
    frames = [("flat", flat, how)]
    natural = np.asarray(vertices, dtype=float)
    if not np.allclose(natural, flat):
        frames.append(("natural", natural, "file_frame"))
    longest = float(max(dims_mm))
    sorted_dims = sorted(float(d) for d in dims_mm)
    for label, verts, how_oriented in frames:
        vox, pitch = voxelise_mesh(verts, faces, fine=fine)
        found, rejects = scaled_variants(vox, rng, rotations=1 if label == "natural" else 2)
        tally.update(rejects)
        for scale, rot, scaled in found:
            # the fine pitch grows by 1 / scale when the grid is re-voxelised by `scale`
            cell_mm = longest / fine / scale
            out.append(
                {
                    "vox": scaled,
                    "scale": scale,
                    "rot": rot,
                    "orientation": label,
                    "oriented_by": how_oriented,
                    "cell_mm": float(cell_mm),
                    "thickness_cells": float(sorted_dims[0] / cell_mm),
                }
            )
    return out, dict(tally)


# ---------------------------------------------------------------------------------------------
# Fusion 360 Gallery assemblies


def parse_transform(coord):
    """4x4 matrix of a Fusion 360 `transform` (origin, x_axis, y_axis, z_axis dicts), in the file's unit."""
    mat = np.eye(4)
    for i, key in enumerate(("x_axis", "y_axis", "z_axis")):
        axis = coord[key]
        mat[:3, i] = [axis["x"], axis["y"], axis["z"]]
    origin = coord["origin"]
    mat[:3, 3] = [origin["x"], origin["y"], origin["z"]]
    return mat


JOINT_KINDS = {
    "RigidJointType": "rigid",
    "RevoluteJointType": "revolute",
    "SliderJointType": "slider",
    "CylindricalJointType": "cylindrical",
    "PinSlotJointType": "pin_slot",
    "PlanarJointType": "planar",
    "BallJointType": "ball",
}
FREE_KINDS = frozenset(JOINT_KINDS.values()) - {"rigid"}


def joint_record(joint):
    """Normalised joint: kind, name, the two occurrence ids and the motion axis / limits when stated."""
    motion = joint.get("joint_motion", {}) or {}
    kind = JOINT_KINDS.get(motion.get("joint_type"), "other")
    record = {
        "kind": kind,
        "name": joint.get("name", ""),
        "one": joint.get("occurrence_one"),
        "two": joint.get("occurrence_two"),
        "as_built": joint.get("type") == "AsBuiltJoint",
    }
    for key in ("rotation_axis", "slide_direction", "pitch_direction", "pitch_direction_one", "pitch_direction_two"):
        if key in motion:
            record[key] = motion[key]
    for key in ("rotation_limits", "slide_limits"):
        if key in motion:
            limits = motion[key]
            record[key] = {
                name: limits.get(name)
                for name in ("minimum_value", "maximum_value", "is_minimum_value_enabled", "is_maximum_value_enabled")
            }
    return record


def parse_assembly(data):
    """Constituent parts of one Fusion 360 assembly JSON (dict).

    Returns (parts, joints, tree): parts is a list of dicts, one per occurrence-held body, with occurrence id,
    world transform (4x4, composed down the tree; file unit),
    occurrence/component/body names, material, physical properties, parent occurrence and depth, the transform;
    joints are joint_record dicts; tree maps occurrence id -> parent occurrence id (None at the top)."""
    occurrences = data.get("occurrences", {}) or {}
    components = data.get("components", {}) or {}
    bodies = data.get("bodies", {}) or {}
    parent = {}
    world = {}

    def walk(node, above, above_world):
        for occ_id, children in (node or {}).items():
            parent[occ_id] = above
            occ = occurrences.get(occ_id, {})
            transform = occ.get("transform")
            local = parse_transform(transform) if transform else np.eye(4)
            world[occ_id] = above_world @ local
            walk(children, occ_id, world[occ_id])

    walk((data.get("tree", {}) or {}).get("root", {}), None, np.eye(4))
    parts = []

    def add(occ_id, occ_name, comp, body_id, body_flags, grounded, world_matrix, depth, above, visible):
        body = bodies.get(body_id, {})
        parts.append(
            {
                "occurrence": occ_id,
                "occurrence_name": occ_name,
                "component_name": comp.get("name", ""),
                "part_number": comp.get("part_number", ""),
                "body": body_id,
                "body_name": body.get("name", ""),
                "material": (body.get("material") or {}).get("name", ""),
                "appearance": (body.get("appearance") or {}).get("name", ""),
                "physical": body.get("physical_properties") or {},
                "parent": above,
                "depth": depth,
                "grounded": grounded,
                "world": world_matrix,
                "visible": visible and bool((body_flags or {}).get("is_visible", True)),
                "obj": body.get("obj"),
            }
        )

    for occ_id, occ in occurrences.items():
        comp = components.get(occ.get("component"), {})
        held = occ.get("bodies") or {}
        body_ids = list(held.keys()) or list(comp.get("bodies") or [])
        depth = 0
        walker = occ_id
        while parent.get(walker) is not None:
            walker = parent[walker]
            depth += 1
        for body_id in body_ids:
            add(occ_id, occ.get("name", ""), comp, body_id, held.get(body_id), bool(occ.get("is_grounded", False)),
                world.get(occ_id, np.eye(4)), depth, parent.get(occ_id), bool(occ.get("is_visible", True)))
    root = data.get("root") or {}
    root_comp = components.get(root.get("component"), {})
    for body_id, flags in (root.get("bodies") or {}).items():
        add(None, "root", root_comp, body_id, flags, False, np.eye(4), 0, None, True)
    joints = [joint_record(j) for j in (data.get("joints", {}) or {}).values()]
    joints += [joint_record(j) for j in (data.get("as_built_joints", {}) or {}).values()]
    return parts, joints, parent


def parse_obj(lines):
    """Vertices and faces of Wavefront .obj text lines (v / f lines only; polygons fan-triangulated; group lines
    and poly lines are ignored). Plain text parsing, nothing from the file is executed."""
    verts = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    return np.array(verts, dtype=float).reshape(-1, 3), np.array(faces, dtype=int).reshape(-1, 3)


def read_obj(path):
    with open(path, "r", errors="replace") as handle:
        return parse_obj(handle)


def whole_assembly_voxels(placed, fine=FINE):
    """Voxelise placed meshes (list of (vertices, faces, part_id)) into one grid with a part id per cell.

    Returns (ids (z, y, x) int16 with -1 empty, pitch). Meshes must already share one frame; the frame is
    z up and the assembly is not re-oriented. Later parts overwrite earlier ones on shared cells."""
    allv = np.concatenate([p[0] for p in placed])
    lo = allv.min(axis=0)
    extent = allv.max(axis=0) - lo
    pitch = float(extent.max()) / fine
    shape = np.ceil(extent / pitch).astype(int) + 2
    ids = -np.ones((shape[2], shape[1], shape[0]), dtype=np.int16)
    for verts, faces, part_id in placed:
        vox, _ = voxelise_mesh(verts - lo + pitch, faces, pitch=pitch)
        zs, ys, xs = np.nonzero(vox)
        keep = (zs < ids.shape[0]) & (ys < ids.shape[1]) & (xs < ids.shape[2])
        ids[zs[keep], ys[keep], xs[keep]] = part_id
    return ids, pitch


def downscale_ids(ids, factor, threshold=goal_library.THRESHOLD):
    """Re-voxelise a part-id grid by `factor` per axis: a coarse cell is filled when `threshold` of the fine
    cells under it are, and takes the most common part id among them. Returns (z, y, x) int16, -1 empty."""
    shape = np.array(ids.shape)
    out_shape = np.maximum(1, np.rint(shape * factor).astype(int))
    out = -np.ones(tuple(out_shape), dtype=np.int16)
    zs, ys, xs = np.nonzero(ids >= 0)
    cz = np.minimum((zs * out_shape[0]) // shape[0], out_shape[0] - 1)
    cy = np.minimum((ys * out_shape[1]) // shape[1], out_shape[1] - 1)
    cx = np.minimum((xs * out_shape[2]) // shape[2], out_shape[2] - 1)
    flat = (cz * out_shape[1] + cy) * out_shape[2] + cx
    count = Counter(flat.tolist())
    votes = {}
    for key, pid in zip(flat.tolist(), ids[zs, ys, xs].tolist()):
        votes.setdefault(key, Counter())[pid] += 1
    box = np.ceil(shape / out_shape).astype(int)
    full = float(np.prod(box))
    for key, n in count.items():
        if n / full >= threshold:
            z, rem = divmod(key, out_shape[1] * out_shape[2])
            y, x = divmod(rem, out_shape[2])
            out[z, y, x] = votes[key].most_common(1)[0][0]
    return out


def assembly_variant(ids, target=250, max_cells=PART_MAX_CELLS):
    """The scale (of goal_library.SCALES) at which the id grid fits 16 per axis, is grounded and has a cell
    count nearest `target` within 4-800; returns (id grid cropped, scale) or (None, None)."""
    best = None
    for scale in goal_library.SCALES:
        scaled = downscale_ids(ids, scale)
        vox = scaled >= 0
        box = goal_library.crop_bbox(vox)
        if box is None or max(box.shape) > goal_library.SIZE:
            continue
        cells = int(vox.sum())
        if cells < goal_library.MIN_CELLS or cells > max_cells:
            continue
        if not goal_library.grounded(goal_library.crop_bbox(vox)):
            continue
        gap = abs(cells - target)
        if best is None or gap < best[0]:
            zs, ys, xs = np.nonzero(vox)
            crop = scaled[zs.min() : zs.max() + 1, ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            best = (gap, crop, scale)
    if best is None:
        return None, None
    return best[1], best[2]
