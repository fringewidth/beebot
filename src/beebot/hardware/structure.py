"""Statics of a voxel structure: which modules can it hold this step?

Modules are rigid cubes (6 degrees of freedom each). Every face contact is a set of springs:

1. a normal spring at each of the face's four corners, compression only (faces press, they do not
   stick by themselves);
1. a magnet spring at the face centre, tension only, for module-to-module contacts: it holds up
   to the connector strength s and the connector fails above it;
1. tangential springs at the corners, active while the contact is closed (some corner pressing or
   the magnet pulling); the contact slides (fails) when its shear exceeds mu times its clamping
   force (corner compression plus s minus the magnet's tension). Terrain contacts are not checked
   for sliding: with gravity the only load, their shear comes from frame action that rigid blocks
   resting on terrain do not need.

Terrain (floor, walls, cheeses) contacts have no magnet. A module that flies also holds its
position by thrust (a stiff spring to its place) as long as the lift asked of it stays within
its thrust; past that it gives its full thrust as a constant force and stops holding.

The structure is solved as a linear elastic system with an active set: corners in tension let go
and come back when pressed, magnets work only in tension. Then the worst failing connector or
sliding contact is removed and the system solved again. A module the remaining contacts cannot
hold shows up as a mechanism (a large displacement against a tiny regularising spring) and is
reported loose; so is everything only it held.

Units: lengths in L, forces in module weights. Physical axes (x, y, z), z up; the grid is (z, y, x).
"""

import numpy as np

MU = 0.5
K_NORMAL = 1.0e4
K_TANGENT = 1.0e3  # soft, so frame action does not load contacts in friction that rigid blocks would not need
K_HOVER = 1.0e4
REGULAR = 1.0e-6
MOVE_LIMIT = 0.5  # a module displaced this far (in L) is not held
TOL = 1e-6
MAX_ROUNDS = 60
RELEASE_ALL = True

FACES = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])  # grid order (dz, dy, dx)
NORMAL, MAGNET, TANGENT = 0, 1, 2


def physical(grid_dir):
    dz, dy, dx = grid_dir
    return np.array([dx, dy, dz], dtype=float)


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def point_jacobian(r):
    """(3, 6): displacement of a point at r from a module centre, u + theta x r."""
    return np.hstack([np.eye(3), -skew(r)])


def face_springs(n):
    """Springs of a contact across face normal n (physical, from module i toward its neighbour):
    (G (S, 12) mapping [q_i, q_j] to each spring's extension, kind (S,), corner (S,) index or -1,
    stiffness (S,))."""
    t1 = np.array([1.0, 0, 0]) if abs(n[0]) < 0.5 else np.array([0, 1.0, 0])
    t2 = np.cross(n, t1)
    rows, kind, corner, stiff = [], [], [], []

    def add(point, d, k, c, s):
        ri, rj = point, point - n
        rows.append(np.concatenate([-d @ point_jacobian(ri), d @ point_jacobian(rj)]))
        kind.append(k)
        corner.append(c)
        stiff.append(s)

    centre = 0.5 * n
    for c, (a, b) in enumerate(((1, 1), (1, -1), (-1, -1), (-1, 1))):
        p = centre + 0.5 * a * t1 + 0.5 * b * t2
        add(p, n, NORMAL, c, K_NORMAL)
        add(p, t1, TANGENT, c, K_TANGENT)
        add(p, t2, TANGENT, c, K_TANGENT)
    add(centre, n, MAGNET, -1, K_NORMAL)
    return np.array(rows), np.array(kind), np.array(corner), np.array(stiff)


SPRINGS = [face_springs(physical(f)) for f in FACES]
N_SPRINGS = len(SPRINGS[0][1])


class Contacts:
    """All contacts of one component: module i, neighbour j (-1 for terrain), grid face index."""

    def __init__(self, i, j, face, strength):
        self.i, self.j, self.face = i, j, face
        self.strength = np.where(j >= 0, strength, 0.0)
        self.g = np.stack([SPRINGS[f][0] for f in face]) if len(face) else np.zeros((0, N_SPRINGS, 12))
        self.kind = SPRINGS[0][1]
        self.k = SPRINGS[0][3]
        self.present = np.ones(len(face), dtype=bool)
        self.active = np.zeros((len(face), N_SPRINGS), dtype=bool)
        self.active[:, self.kind == NORMAL] = True
        self.active[:, self.kind == TANGENT] = True
        self.active[:, self.kind == MAGNET] = (j >= 0)[:, None] & (self.strength > 0)[:, None]

    def dofs(self):
        """(C, 12) global dof indices; terrain halves point at a dummy dof dropped later."""
        base = np.arange(6)
        di = 6 * self.i[:, None] + base
        dj = np.where(self.j[:, None] >= 0, 6 * self.j[:, None] + base, -1)
        return np.concatenate([di, dj], axis=1)


def assemble(n, contacts, hover):
    dof = 6 * n
    k = np.zeros((dof + 1, dof + 1))  # the last row/column is the dummy terrain dof
    live = contacts.present
    if live.any():
        weights = contacts.active[live] * contacts.k  # (C, S)
        g = contacts.g[live]
        blocks = np.einsum("cs,csi,csj->cij", weights, g, g)
        idx = contacts.dofs()[live]
        idx = np.where(idx < 0, dof, idx)
        np.add.at(k, (idx[:, :, None], idx[:, None, :]), blocks)
    k = k[:dof, :dof]
    k[np.diag_indices(dof)] += REGULAR
    for a in np.nonzero(hover)[0]:
        k[6 * a + np.arange(6), 6 * a + np.arange(6)] += K_HOVER
    return k


def update_active(contacts, stretch, force, normal, magnet, tangent):
    """One active-set update; True when anything changed. Pressed corners that were open close
    first; only when none do, the most stretched open-in-tension corner of each contact lets go
    (one per contact per round, which keeps the iteration from oscillating)."""
    live = contacts.present
    was = contacts.active.copy()
    corners = contacts.active[:, normal].copy()
    closing = ~corners & (stretch[:, normal] < -TOL / K_NORMAL) & live[:, None]
    if closing.any():
        corners |= closing
    elif RELEASE_ALL:
        corners &= ~((force[:, normal] > TOL) & live[:, None])
    else:
        pulling = np.where(corners & live[:, None], force[:, normal], -np.inf)
        worst = pulling.argmax(1)
        rows = np.nonzero(pulling.max(1) > TOL)[0]
        corners[rows, worst[rows]] = False
    contacts.active[:, normal] = corners
    has_magnet = (contacts.j >= 0) & (contacts.strength > 0)
    contacts.active[:, magnet] = (has_magnet & (stretch[:, magnet][:, 0] > -TOL / K_NORMAL))[:, None]
    closed = corners.any(1) | contacts.active[:, magnet][:, 0]
    contacts.active[:, tangent] = closed[:, None]
    return bool(((contacts.active != was) & live[:, None]).any())


def holds(n, contacts, load, hover, thrust):
    """Bool per module: held by the structure. load (n,) downward load in weights; hover (n,)
    bool modules holding position by thrust (each up to `thrust` weights)."""
    held = np.ones(n, dtype=bool)
    hover = hover.copy()
    lift = np.zeros(n)
    seen = set()
    for _ in range(MAX_ROUNDS):
        contacts.present &= held[contacts.i] & np.where(contacts.j >= 0, held[np.maximum(contacts.j, 0)], True)
        rhs = np.zeros(6 * n)
        rhs[6 * np.arange(n) + 2] = -(load - lift) * held
        u = np.linalg.solve(assemble(n, contacts, hover & held), rhs)
        q = u.reshape(n, 6)
        both = np.concatenate([q[contacts.i], np.where(contacts.j[:, None] >= 0, q[np.maximum(contacts.j, 0)], 0.0)], axis=1)
        stretch = np.einsum("csi,ci->cs", contacts.g, both)  # (C, S) extension of each spring
        force = stretch * contacts.k  # tension positive
        normal = contacts.kind == NORMAL
        magnet = contacts.kind == MAGNET
        tangent = contacts.kind == TANGENT
        far = np.abs(q[:, :3]).max(1) > MOVE_LIMIT
        if update_active(contacts, stretch, force, normal, magnet, tangent):
            state = contacts.active.tobytes() + contacts.present.tobytes()
            if state not in seen:  # a repeated state is a cycle: settle it as below
                seen.add(state)
                continue
        seen.clear()
        corners = contacts.active[:, normal]
        if (far & held).any():
            held &= ~far
            continue
        if hover.any():
            reaction = -K_HOVER * q[:, 2]
            over = hover & held & (reaction > thrust + TOL)
            if over.any():
                worst = np.argmax(np.where(over, reaction, -np.inf))
                hover[worst] = False
                lift[worst] = thrust
                continue
        pull = np.where(contacts.active[:, magnet][:, 0], force[:, magnet][:, 0], 0.0)
        press = -np.where(corners, force[:, normal], 0.0).sum(1)
        shear_vec = np.zeros((len(contacts.face), 3))
        tang_idx = np.nonzero(tangent)[0]
        for f in range(6):
            sel = contacts.face == f
            if sel.any():
                d = -contacts.g[sel][:, tang_idx, :3]  # each tangential spring's direction (its row on module i's translation is -d)
                shear_vec[sel] = np.einsum("cs,csk->ck", np.where(contacts.active[sel][:, tang_idx], force[sel][:, tang_idx], 0.0), d)
        shear = np.linalg.norm(shear_vec, axis=1)
        clamp = press + np.maximum(contacts.strength - pull, 0.0)
        excess = np.maximum((pull - contacts.strength) / np.maximum(contacts.strength, 1.0), (shear - MU * clamp) / np.maximum(contacts.strength, 1.0))
        excess = np.where(contacts.present & (contacts.j >= 0), excess, -np.inf)  # terrain contacts do not slide here
        if excess.size and excess.max() > TOL:
            contacts.present[int(np.argmax(excess))] = False
            continue
        return held
    return held


def column_supported(occupied, solid):
    """(D, H, W) bool: modules with an unbroken column of modules under them down to terrain."""
    d = occupied.shape[0]
    column = np.zeros_like(occupied)
    for z in range(d):
        below = (solid[z - 1] | column[z - 1]) if z > 0 else np.ones(occupied.shape[1:], dtype=bool)
        column[z] = occupied[z] & below
    return column


def contacts_of(occupied, solid):
    """Module positions (n, 3), and contacts: module i, neighbour j (-1 terrain), face index."""
    pos = np.argwhere(occupied)
    dims = np.array(occupied.shape)
    index = -np.ones(occupied.shape, dtype=int)
    index[tuple(pos.T)] = np.arange(len(pos))
    ii, jj, ff = [], [], []
    for f, step in enumerate(FACES):
        there = pos + step
        inside = (there >= 0).all(1) & (there < dims).all(1)
        clipped = np.clip(there, 0, dims - 1)
        other = np.where(inside, index[tuple(clipped.T)], -1)
        terrain = np.where(inside, solid[tuple(clipped.T)], there[:, 0] < 0)
        pair = (other >= 0) & (f % 2 == 0)  # each module pair once
        for keep, j in ((pair, other), (terrain, np.full(len(pos), -1))):
            ii.append(np.nonzero(keep)[0])
            jj.append(j[keep])
            ff.append(np.full(int(keep.sum()), f))
    return pos, np.concatenate(ii), np.concatenate(jj), np.concatenate(ff)


def labels(n, i, j):
    """Connected components of modules under module-to-module contacts."""
    parent = np.arange(n)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(i[j >= 0], j[j >= 0]):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([root(a) for a in range(n)])


def loose_modules(occupied, solid, extra, strength, flies, thrust, memory=None):
    """(D, H, W) bool: modules the structure cannot hold. occupied, solid (D, H, W) bool; extra
    (D, H, W) float, load resting on each module in module weights. memory: a dict carried from
    the previous step of the same scene, holding which corners of each contact were pressing; the
    active set starts from it (most contacts persist between steps), and it is updated in place."""
    loose = np.zeros_like(occupied)
    previous = dict(memory) if memory is not None else {}
    if memory is not None:
        memory.clear()
    pos, i, j, face = contacts_of(occupied, solid)
    n = len(pos)
    if n == 0:
        return loose
    stands = column_supported(occupied, solid)[tuple(pos.T)]
    load = 1.0 + extra[tuple(pos.T)]
    label = labels(n, i, j)
    for c in np.unique(label):
        members = np.nonzero(label == c)[0]
        if stands[members].all():
            continue  # columns on terrain: compression only
        if flies and (load[members] <= thrust).all():
            continue  # every module hovers under its own load
        local = -np.ones(n, dtype=int)
        local[members] = np.arange(len(members))
        sel = label[i] == c
        contacts = Contacts(local[i[sel]], np.where(j[sel] >= 0, local[np.maximum(j[sel], 0)], -1), face[sel], strength)
        keys = [(*pos[a], f) for a, f in zip(i[sel], face[sel])]
        normal = contacts.kind == NORMAL
        for row, key in enumerate(keys):
            if key in previous:
                contacts.active[row, normal] = previous[key]
        held = holds(len(members), contacts, load[members], np.full(len(members), flies), thrust)
        loose[tuple(pos[members[~held]].T)] = True
        if memory is not None:
            for row, key in enumerate(keys):
                memory[key] = contacts.active[row, normal].copy()
    return loose
