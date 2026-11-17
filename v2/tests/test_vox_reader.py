"""MagicaVoxel and Slab6 readers on small synthetic files."""

import struct

import numpy as np
import pytest

from beebot.vla.voxel import vox


def chunk(tag, body, children=b""):
    return tag + struct.pack("<ii", len(body), len(children)) + body + children


def string(text):
    raw = text.encode()
    return struct.pack("<i", len(raw)) + raw


def vox_dict(items):
    out = struct.pack("<i", len(items))
    for key, value in items.items():
        out += string(key) + string(value)
    return out


def make_vox(models, palette=None, names=None):
    """models: list of ((sx, sy, sz), [(x, y, z, i)]); names: scene graph transform names per model."""
    body = b""
    for (sx, sy, sz), voxels in models:
        body += chunk(b"SIZE", struct.pack("<iii", sx, sy, sz))
        xyzi = b"".join(struct.pack("<BBBB", *v) for v in voxels)
        body += chunk(b"XYZI", struct.pack("<i", len(voxels)) + xyzi)
    if names:
        # node 0: transform -> group 1 -> per model: transform (named) -> shape
        kids = list(range(2, 2 + 2 * len(models), 2))
        body += chunk(b"nTRN", struct.pack("<i", 0) + vox_dict({}) + struct.pack("<iiii", 1, -1, -1, 0) + struct.pack("<i", 0))
        body += chunk(b"nGRP", struct.pack("<i", 1) + vox_dict({}) + struct.pack("<i", len(kids)) + struct.pack(f"<{len(kids)}i", *kids))
        for k, name in enumerate(names):
            trn = 2 + 2 * k
            body += chunk(b"nTRN", struct.pack("<i", trn) + vox_dict({"_name": name}) + struct.pack("<iiii", trn + 1, -1, -1, 0) + struct.pack("<i", 0))
            body += chunk(b"nSHP", struct.pack("<i", trn + 1) + vox_dict({}) + struct.pack("<i", 1) + struct.pack("<i", k) + vox_dict({}))
    if palette is not None:
        body += chunk(b"RGBA", palette.astype(np.uint8).tobytes())
    return b"VOX " + struct.pack("<i", 150) + chunk(b"MAIN", b"", body)


def test_single_model_indices_and_palette():
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[:, 3] = 255
    palette[0, :3] = (255, 0, 0)  # colour index 1
    palette[1, :3] = (0, 255, 0)  # colour index 2
    data = make_vox([((2, 3, 4), [(0, 0, 0, 1), (1, 2, 3, 2)])], palette=palette)
    (name, arr, rgb), = vox.models(data, fallback="thing.vox")
    assert name == "thing.vox"
    assert arr.shape == (4, 3, 2)  # (z, y, x)
    assert arr[0, 0, 0] == 1 and arr[3, 2, 1] == 2 and (arr > 0).sum() == 2
    assert tuple(rgb[arr[0, 0, 0] - 1]) == (255, 0, 0)
    assert tuple(rgb[arr[3, 2, 1] - 1]) == (0, 255, 0)


def test_scene_graph_names_and_skipped_empty_model():
    data = make_vox([((1, 1, 1), [(0, 0, 0, 1)]), ((2, 2, 2), []), ((1, 1, 2), [(0, 0, 1, 3)])], names=["Shop - Base", "empty", "Barrel"])
    found = vox.models(data)
    assert [n for n, _, _ in found] == ["Shop - Base", "Barrel"]
    assert found[1][1].shape == (2, 1, 1)


def test_bad_inputs_raise_not_crash():
    with pytest.raises(vox.VoxError):
        vox.parse(b"not a vox file at all, just text")
    good = make_vox([((1, 1, 1), [(0, 0, 0, 1)])])
    with pytest.raises(vox.VoxError):
        vox.parse(good[:28])  # truncated chunk
    with pytest.raises(vox.VoxError):
        vox.slab6(b"\x01\x00\x00\x00" * 3 + b"xx")


def test_out_of_range_voxels_are_ignored():
    data = make_vox([((2, 2, 2), [(0, 0, 0, 1), (5, 5, 5, 1)])])
    (_, arr, _), = vox.models(data)
    assert int((arr > 0).sum()) == 1


def test_slab6_orientation_and_palette():
    sx, sy, sz = 2, 2, 3
    body = np.full((sx, sy, sz), 255, dtype=np.uint8)
    body[1, 0, 2] = 7  # slab6 z is down: z = 2 is the lowest layer
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[7] = (63, 0, 32)  # 6-bit values
    data = struct.pack("<iii", sx, sy, sz) + body.tobytes() + palette.tobytes()
    (_, arr, rgb), = vox.slab6(data)
    assert arr.shape == (3, 2, 2)
    assert arr[0, 0, 1] == 8  # lowest layer is z = 0 (up), y = 0, x = 1; index is byte + 1
    assert (arr > 0).sum() == 1
    assert tuple(rgb[arr[0, 0, 1] - 1]) == (254, 2, 130)
    assert vox.any_vox(data)[0][1].shape == (3, 2, 2)


def test_readable():
    assert vox.readable("Watch_Tower-02.vox") == "watch tower"
    assert vox.readable("barrelTop.vox") == "barrel top"
