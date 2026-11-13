"""Reader for MagicaVoxel .vox files (versions 150 and 200), pure Python and numpy.

`parse(data)` reads bytes (never executes anything) into models (size, voxel list), the palette and the scene
graph names; `models(data)` returns one named item per model: a (z, y, x) uint8 array of palette indices
(0 empty, 1..255 colour) and the (256, 3) uint8 RGB palette. MagicaVoxel is z up, x right, y forward, so a
model is returned as an array indexed (z, y, x). Models of a multi-model file are named by the scene graph
transform node (`_name`) that reaches them, else by the file name and the model index.
"""

import struct

import numpy as np

DEFAULT_PALETTE = None  # files without an RGBA chunk use MagicaVoxel's default; we fall back to grey ramps


class VoxError(ValueError):
    pass


def _dict(data, pos):
    (count,) = struct.unpack_from("<i", data, pos)
    pos += 4
    out = {}
    for _ in range(count):
        (klen,) = struct.unpack_from("<i", data, pos)
        pos += 4
        key = data[pos : pos + klen].decode("utf8", "replace")
        pos += klen
        (vlen,) = struct.unpack_from("<i", data, pos)
        pos += 4
        out[key] = data[pos : pos + vlen].decode("utf8", "replace")
        pos += vlen
    return out, pos


def parse(data):
    """bytes -> {"version", "sizes": [(x, y, z)], "voxels": [(n, 4) uint8 x y z i], "palette": (256, 4) or None,
    "nodes": {id: ("trn"|"grp"|"shp", ...)}}. Raises VoxError on bad input."""
    if len(data) < 20 or data[:4] != b"VOX ":
        raise VoxError("not a vox file")
    version = struct.unpack_from("<i", data, 4)[0]
    if data[8:12] != b"MAIN":
        raise VoxError("no MAIN chunk")
    content, children = struct.unpack_from("<ii", data, 12)
    pos = 20 + content
    if children < 0 or pos + children > len(data):
        raise VoxError("truncated file")
    end = pos + children
    sizes, voxels, nodes = [], [], {}
    palette = None
    while pos + 12 <= end:
        tag = data[pos : pos + 4]
        clen, ccl = struct.unpack_from("<ii", data, pos + 4)
        body = pos + 12
        if clen < 0 or body + clen > len(data):
            raise VoxError("truncated chunk")
        if tag == b"SIZE":
            sizes.append(struct.unpack_from("<iii", data, body))
        elif tag == b"XYZI":
            (n,) = struct.unpack_from("<i", data, body)
            if n < 0 or body + 4 + 4 * n > len(data):
                raise VoxError("bad XYZI")
            voxels.append(np.frombuffer(data, dtype=np.uint8, count=4 * n, offset=body + 4).reshape(n, 4))
        elif tag == b"RGBA":
            palette = np.frombuffer(data, dtype=np.uint8, count=1024, offset=body).reshape(256, 4)
        elif tag == b"nTRN":
            node_id = struct.unpack_from("<i", data, body)[0]
            attrs, p = _dict(data, body + 4)
            child = struct.unpack_from("<i", data, p)[0]
            nodes[node_id] = ("trn", attrs, child)
        elif tag == b"nGRP":
            node_id = struct.unpack_from("<i", data, body)[0]
            attrs, p = _dict(data, body + 4)
            (n,) = struct.unpack_from("<i", data, p)
            kids = list(struct.unpack_from(f"<{n}i", data, p + 4))
            nodes[node_id] = ("grp", attrs, kids)
        elif tag == b"nSHP":
            node_id = struct.unpack_from("<i", data, body)[0]
            attrs, p = _dict(data, body + 4)
            (n,) = struct.unpack_from("<i", data, p)
            p += 4
            ids = []
            for _ in range(n):
                ids.append(struct.unpack_from("<i", data, p)[0])
                _, p = _dict(data, p + 4)
            nodes[node_id] = ("shp", attrs, ids)
        pos = body + clen + ccl
    if len(voxels) != len(sizes):
        raise VoxError("SIZE and XYZI counts differ")
    return {"version": version, "sizes": sizes, "voxels": voxels, "palette": palette, "nodes": nodes}


def model_names(parsed):
    """{model index: name} from the scene graph (the nearest named transform above each shape node)."""
    nodes = parsed["nodes"]
    names = {}

    def walk(node_id, name):
        node = nodes.get(node_id)
        if node is None:
            return
        if node[0] == "trn":
            name = node[1].get("_name", "") or name
            walk(node[2], name)
        elif node[0] == "grp":
            for kid in node[2]:
                walk(kid, name)
        else:
            for index in node[2]:
                if name and index not in names:
                    names[index] = name

    if 0 in nodes:
        walk(0, "")
    return names


def models(data, fallback="model"):
    """[(name, indices (z, y, x) uint8, rgb palette (256, 3) uint8 where row i is colour index i + 1)]; empty
    models are skipped."""
    parsed = parse(data)
    palette = parsed["palette"]
    if palette is None:
        ramp = np.linspace(40, 235, 256).astype(np.uint8)
        palette = np.stack([ramp, ramp, ramp, np.full(256, 255, np.uint8)], axis=1)
    rgb = np.ascontiguousarray(palette[:, :3])
    names = model_names(parsed)
    out = []
    for index, ((sx, sy, sz), xyzi) in enumerate(zip(parsed["sizes"], parsed["voxels"])):
        if len(xyzi) == 0 or min(sx, sy, sz) <= 0:
            continue
        arr = np.zeros((sz, sy, sx), dtype=np.uint8)
        x, y, z, i = (xyzi[:, k].astype(int) for k in range(4))
        keep = (x < sx) & (y < sy) & (z < sz) & (i > 0)
        arr[z[keep], y[keep], x[keep]] = i[keep]
        if not arr.any():
            continue
        out.append((names.get(index) or (fallback if len(parsed["sizes"]) == 1 else f"{fallback} {index}"), arr, rgb))
    return out


def slab6(data, fallback="model"):
    """A Slab6 .vox (some packs use it: int32 x, y, z sizes, x*y*z palette bytes with 255 = empty, 256 x 6-bit RGB
    palette; z is down) as [(name, indices (z, y, x), rgb palette)]. Palette index i + 1 is stored for byte i."""
    if len(data) < 12:
        raise VoxError("short slab6")
    sx, sy, sz = struct.unpack_from("<iii", data, 0)
    if min(sx, sy, sz) <= 0 or sx * sy * sz + 12 + 768 != len(data):
        raise VoxError("not a slab6 file")
    body = np.frombuffer(data, dtype=np.uint8, count=sx * sy * sz, offset=12).reshape(sx, sy, sz)
    palette = np.frombuffer(data, dtype=np.uint8, count=768, offset=12 + sx * sy * sz).reshape(256, 3)
    rgb = np.ascontiguousarray(np.minimum(palette.astype(np.int32) * 4 + 2, 255).astype(np.uint8))
    arr = np.where(body == 255, 0, body.astype(np.int32) + 1)
    arr = arr[:, :, ::-1].transpose(2, 1, 0)  # (x, y, z down) -> (z up, y, x)
    arr = np.ascontiguousarray(np.minimum(arr, 255).astype(np.uint8))
    return [(fallback, arr, rgb)] if arr.any() else []


def any_vox(data, fallback="model"):
    """MagicaVoxel or Slab6, decided by the header."""
    if data[:4] == b"VOX ":
        return models(data, fallback=fallback)
    return slab6(data, fallback=fallback)


def readable(name):
    """A file or model name as words: 'Watch_Tower-02.vox' -> 'watch tower'. Digits and extensions are dropped."""
    import re

    stem = re.sub(r"\.(vox|zip)$", "", name.strip(), flags=re.I)
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    stem = re.sub(r"[_\-\.\s]+", " ", stem)
    stem = re.sub(r"\b\d+\b", " ", stem)
    return re.sub(r"\s+", " ", stem).strip().lower()
