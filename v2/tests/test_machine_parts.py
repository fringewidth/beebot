"""Machine parts: parsing and voxelising on small synthetic inputs (no downloads)."""

import numpy as np

from beebot.vla.voxel import goal_library, machine_parts as mp


def box_mesh(size=(2.0, 1.0, 0.5), origin=(0.0, 0.0, 0.0)):
    sx, sy, sz = size
    ox, oy, oz = origin
    verts = np.array([[ox + x * sx, oy + y * sy, oz + z * sz] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = []
    for a, b, c, d in quads:
        faces += [(a, b, c), (a, c, d)]
    return verts, np.array(faces)


def test_generic_names():
    for name in ["Body12", "Component3:1", "body 1", "Untitled", "", "42", "0f3a9c1e-77aa", "Part1", "root", "prt0002", "asm0002"]:
        assert mp.name_is_generic(name), name
    for name in ["steering upright", "Hex bolt M6", "Gear 20T", "Lid"]:
        assert not mp.name_is_generic(name), name


def test_shape_class():
    assert mp.shape_class((10, 10, 10)) == "cube"
    assert mp.shape_class((1, 20, 22)) == "plate"
    assert mp.shape_class((2, 3, 40)) == "rod"
    assert mp.shape_class((1, 5, 40)) == "strip"
    assert mp.shape_class((4, 6, 8)) == "block"


def test_rotation_aligning():
    rot = mp.rotation_aligning([1, 0, 0], [0, 0, -1])
    assert np.allclose(rot @ np.array([1.0, 0, 0]), [0, 0, -1])
    assert np.allclose(mp.rotation_aligning([0, 0, 1], [0, 0, -1]) @ [0, 0, 1], [0, 0, -1])


def test_orient_flat_rests_largest_face_down():
    verts, faces = box_mesh((0.5, 1.0, 2.0))  # largest face is the 1 x 2 one, normal along x
    flat, how = mp.orient_flat(verts, faces)
    extent = flat.max(axis=0) - flat.min(axis=0)
    assert how == "face"
    assert np.isclose(extent[2], 0.5)


def test_voxelise_box_filled():
    verts, faces = box_mesh((2.0, 1.0, 0.5))
    vox, pitch = mp.voxelise_mesh(verts, faces, fine=16)
    assert vox.shape[2] >= vox.shape[1] >= vox.shape[0]
    assert vox.sum() >= 0.8 * np.prod(vox.shape)  # filled, not a shell


def test_part_variants_accepted_and_grounded():
    verts, faces = box_mesh((20.0, 20.0, 20.0))
    found, tally = mp.part_variants(verts, faces, (20.0, 20.0, 20.0), np.random.default_rng(0), fine=32)
    assert found
    for item in found:
        assert goal_library.MIN_CELLS <= item["vox"].sum() <= mp.PART_MAX_CELLS
        assert max(item["vox"].shape) <= goal_library.SIZE
        assert goal_library.grounded(item["vox"])
        assert item["cell_mm"] > 0


def test_thin_plate_thickness_below_one_cell():
    verts, faces = box_mesh((40.0, 40.0, 1.0))
    found, _ = mp.part_variants(verts, faces, (40.0, 40.0, 1.0), np.random.default_rng(0), fine=32)
    assert found
    assert min(item["thickness_cells"] for item in found) < 1.0


def test_parse_assembly_and_joints():
    data = {
        "tree": {"root": {"o1": {"o2": {}}}},
        "occurrences": {
            "o1": {"name": "arm", "component": "c1", "bodies": {}, "is_grounded": True},
            "o2": {"name": "pin", "component": "c2", "bodies": {}},
        },
        "components": {
            "c1": {"name": "arm", "part_number": "A1", "bodies": ["b1"]},
            "c2": {"name": "Component2", "part_number": "Untitled", "bodies": ["b2"]},
        },
        "bodies": {
            "b1": {"name": "Body1", "obj": "b1.obj", "material": {"name": "Steel"}},
            "b2": {"name": "pin", "obj": "b2.obj"},
        },
        "joints": {
            "j1": {
                "name": "Rev1",
                "occurrence_one": "o1",
                "occurrence_two": "o2",
                "joint_motion": {"joint_type": "RevoluteJointType", "rotation_axis": "ZAxisJointDirection"},
            }
        },
        "as_built_joints": {
            "j2": {"name": "Rigid1", "occurrence_one": "o1", "occurrence_two": "o2", "joint_motion": {"joint_type": "RigidJointType"}}
        },
    }
    parts, joints, parent = mp.parse_assembly(data)
    assert {p["body"] for p in parts} == {"b1", "b2"}
    assert parent == {"o1": None, "o2": "o1"}
    by_body = {p["body"]: p for p in parts}
    assert by_body["b2"]["parent"] == "o1" and by_body["b2"]["depth"] == 1
    assert by_body["b1"]["material"] == "Steel" and by_body["b1"]["grounded"]
    kinds = sorted(j["kind"] for j in joints)
    assert kinds == ["revolute", "rigid"]
    assert "revolute" in mp.FREE_KINDS and "rigid" not in mp.FREE_KINDS


def test_read_obj(tmp_path):
    path = tmp_path / "t.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\ng face 1\nf 1/1 2/2 3/3 4/4\nl 1 2\n")
    verts, faces = mp.read_obj(path)
    assert verts.shape == (4, 3) and faces.shape == (2, 3)


def test_whole_assembly_ids_and_variant():
    a = box_mesh((10.0, 10.0, 4.0))
    b = box_mesh((4.0, 4.0, 6.0), origin=(3.0, 3.0, 4.0))
    ids, pitch = mp.whole_assembly_voxels([(a[0], a[1], 0), (b[0], b[1], 1)], fine=32)
    assert set(np.unique(ids)) >= {-1, 0, 1}
    crop, scale = mp.assembly_variant(ids)
    assert crop is not None
    assert set(np.unique(crop[crop >= 0])) == {0, 1}
    assert goal_library.grounded(crop >= 0)
