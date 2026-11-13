"""Joint strength: can a voxel structure of hinge cubes hold itself (and a load) up?

The voxel world's rule (world.supported) says a cube stands when it is face-connected to the
floor, terrain or an object. That is optimistic: a long cantilever, a wide span or a tall tower
under a side load opens at its weakest joint. This module is the check: for cubes on a lattice
under gravity, plus optional loads on given cells, it finds the largest factor by which the loads
can be scaled before some joint must open or slide (the load factor; the structure stands when it
is at least 1), and when it is below 1, which cubes move in the collapse and which joints open.

Model (rigid-block limit analysis, the lower-bound theorem: the structure stands if some set of
joint forces balances every cube within every joint's limits):

1. Cubes are rigid. The floor (below z = 0), terrain cells (`solid`) and anchored cubes
   (`anchored`, e.g. a wall in a test) are fixed.
1. Magnets. Each face carries the hinge cube's magnets (cube.py, cube_swarm.py): two per edge, a
   quarter of the edge from its middle, inset by e = d0 / (2 sqrt 3) along both face normals, and
   one at the face centre. Twin magnets across a closed face sit exactly at the peak of the force
   law (cube.Magnets), so a closed joint is clamped by every magnet at its full pull: 8 edge pairs
   of `edge_pull` and the centre pair of `face_pull` (module weights). Any opening lowers the pull
   (the law falls past its peak), so a joint fails as soon as it must open: these are constant
   forces, not capacities that load up. Cubes that share only an edge are pulled together by that
   edge's two magnet pairs at `edge_share` of a twin's pull (the pair sits at 0.82 d0, not at the
   peak). The steel floor pulls each bottom magnet with `floor_pull` of a twin; terrain with
   `terrain_pull`.
1. Contacts. A face contact presses anywhere on the face: compression at the four corners, each
   with a friction pyramid (|f1| + |f2| <= mu c, inscribed in the Coulomb cone; it also resists
   torsion). An edge-only contact presses at the edge's two ends in the wedge between the two
   cubes' face normals, with friction along the edge: a pin, no moment about the edge.
1. Unknowns: the corner forces, one per pyramid edge, all >= 0, and the load factor. Equilibrium
   of every free cube (force and moment, 6 rows each); maximise the factor (capped at `cap`).
   Its dual is the collapse mechanism (virtual velocities of the cubes), so the cubes that move
   and the joints that open come out of the same solve.

What it leaves out: magnets' resistance to sliding (only friction counts; conservative), the
softness of contacts, dynamic loads (moves, impacts), joint wear and placement gaps, and that the
lower-bound theorem assumes ductile joints: here a joint that opens loses its pull, so in a
structure that must redistribute (a span) the true limit can be a little lower than the bound.
Known failure: a flat beam wedged between immovable supports (terrain, anchored cubes) arches as
rigid blocks and never fails here; in MuJoCo it fails at about 290 / w^2 g for span w >= 25
(HARDWARE.md, "Joint strength"). Validated against MuJoCo otherwise to a median 5 % in the
failure factor (runs/handoff/joint-strength.md).

Units: lengths in L, forces in module weights (m g). Cells are grid (z, y, x) with z up and the
floor layer at z = 0; physical vectors are (x, y, z).

The LP is solved by a batched primal-dual interior-point method (Mehrotra predictor-corrector) in
torch, float64 on the CPU: no scipy. Many structures go in one batch (padded); each splits into
its connected components first. A structure the solver does not converge on (some large or
slender ones) gets a certified lower bound and Result.converged False. Speed (4 threads): about
12 ms a structure of 10 cubes, 75 ms of 30, 3 s of 100.

    from beebot.hardware.joint_strength import check
    r = check([(1, 0, x) for x in range(8)], anchored=[(1, 0, -1), (0, 0, -1)])
    r.factor, r.stands, r.moving, r.opening  # cantilever of 8 from a wall: 0.567, False, all 8 cubes, the root face
"""

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from beebot.hardware.cube import SHAPE_PEAK

AXES = np.eye(3)
REG = 1e-11  # regularisation of the normal matrix, relative to its largest diagonal entry
EPS = 1e-4  # load factors within this of 1 count as 1 (the solver's accuracy is about 1e-6)


def _shape(x):
    return x / (1 + x * x) ** 2


@dataclass(frozen=True)
class Joints:
    """The joints' parameters. Defaults: the as-built hinge cube (cube_swarm.DESIGN.cube(): 5 cm,
    150 g, edge_pull 4.5, face_pull 0.3, magnet_range 0.03, steel floor 0.3)."""

    edge_pull: float = 4.5  # module weights: one twin pair of edge magnets
    face_pull: float = 0.3  # module weights: the twin pair of face-centre magnets
    magnet_range: float = 0.03  # d0 / L
    floor_pull: float = 0.3  # a steel floor's pull on each magnet, relative to a twin's
    terrain_pull: float = 0.3  # terrain's (walls, objects), relative to a twin's (steel like the floor)
    mu: float = 0.5  # friction, cube on cube and cube on floor (MuJoCo jitters it in 0.3-0.7)
    edges: bool = True  # edge-only contacts (pins with their two magnet pairs)
    cap: float = 10.0  # the largest load factor reported (a filter needs no more; the solver is slowest on structures far stronger than the cap)

    @property
    def inset(self):
        return self.magnet_range / math.sqrt(3) / 2

    @property
    def edge_share(self):
        """Pull of an edge-only magnet pair relative to twins at a face (cube.Magnets' law)."""
        r = 2 * math.sqrt(2) * self.inset / self.magnet_range
        return _shape(r) / SHAPE_PEAK

    def moment(self):
        """Moment a face joint holds about one of its edges by its magnets alone (m g L): 4
        edge_pull + 0.5 face_pull."""
        return 4 * self.edge_pull + 0.5 * self.face_pull

    @classmethod
    def from_cube(cls, cube, **options):
        return cls(edge_pull=cube.edge_pull, face_pull=cube.face_pull, magnet_range=cube.magnet_range, floor_pull=cube.floor_pull, terrain_pull=cube.floor_pull, mu=cube.friction, **options)


def _frame(n):
    t1 = AXES[0] if abs(n[0]) < 0.5 else AXES[1]
    t2 = np.cross(n, t1)
    return t1, t2


@dataclass
class Interfaces:
    """Generators and constant forces of one structure. Bodies are free cubes 0..n-1 and -1 (fixed).
    A generator k is a force along d[k] at p[k] on body b[k] and its opposite on body a[k]."""

    centres: np.ndarray  # (n, 3) physical
    a: np.ndarray  # (K,) body or -1
    b: np.ndarray  # (K,)
    p: np.ndarray  # (K, 3)
    d: np.ndarray  # (K, 3)
    joint: np.ndarray  # (K,) index into pairs
    pairs: list  # (a, b, kind) per joint, kind "face" or "edge"; a = -1 fixed
    fixed_force: np.ndarray  # (n, 6) constant wrench on each free cube (magnets), about its centre


def _face_template(mu):
    """Per face normal (6, signed axes): corner offsets (16, 3) from the face centre and generator
    directions (16, 3): four pyramid edges at each of the four corners."""
    out = {}
    for axis in range(3):
        for sign in (-1, 1):
            nrm = sign * AXES[axis]
            t1, t2 = _frame(nrm)
            offs, dirs = [], []
            for c1 in (-0.5, 0.5):
                for c2 in (-0.5, 0.5):
                    for t in (t1, -t1, t2, -t2):
                        offs.append(c1 * t1 + c2 * t2)
                        dirs.append(nrm + mu * t)
            out[(axis, sign)] = (np.array(offs), np.array(dirs))
    return out


def _edge_template(u, v, mu):
    """Edge-only contact pushing along u and v: end offsets (8, 3) from the edge's middle and
    generator directions (8, 3)."""
    e = np.abs(np.cross(u, v))
    offs, dirs = [], []
    for end in (-0.5, 0.5):
        for nrm in (u, v):
            for t in (e, -e):
                offs.append(end * e)
                dirs.append(nrm + mu * t)
    return np.array(offs), np.array(dirs)


def interfaces(cells, solid=(), anchored=(), strength=None, joints=Joints()):
    """The joints of free cubes at `cells` ((n, 3) grid (z, y, x)) with each other, with the floor,
    with terrain cells `solid` and with fixed cubes `anchored` (both lists of grid cells).
    strength: (n,) per-cube magnet strength factors (and for anchored cubes, appended).

    A face's nine magnets are symmetric about its centre and pull along its normal, so for
    equilibrium they are one force of (8 edge_pull + face_pull) x pull at the centre; an edge
    contact's two pairs likewise one force at the edge's middle."""
    cells = np.asarray(cells, dtype=int).reshape(-1, 3)
    anchored = np.asarray(anchored, dtype=int).reshape(-1, 3)
    n = len(cells)
    xyz = cells[:, ::-1].astype(float)
    s_all = np.ones(n + len(anchored)) if strength is None else np.asarray(strength, dtype=float)
    index = {tuple(c): i for i, c in enumerate(cells)}
    fixed_cube = {tuple(c): n + k for k, c in enumerate(anchored)}
    terrain = {tuple(c) for c in np.asarray(solid, dtype=int).reshape(-1, 3)}
    face_pull = 8 * joints.edge_pull + joints.face_pull
    template = _face_template(joints.mu)
    pairs = []
    blocks = []  # (a, b, offsets + point (k, 3), directions (k, 3), joint)
    mag_a, mag_b, mag_p, mag_f = [], [], [], []  # magnet resultants: force mag_f (3,) on b at mag_p, opposite on a

    def face(a, b, centre, axis, sign, pull):
        j = len(pairs)
        pairs.append((a, b, "face"))
        offs, dirs = template[(axis, sign)]
        blocks.append((a, b, centre + offs, dirs, j))
        if pull > 0:
            mag_a.append(a)
            mag_b.append(b)
            mag_p.append(centre)
            mag_f.append(-sign * pull * face_pull * AXES[axis])

    for i, c in enumerate(cells):
        here = xyz[i]
        if c[0] == 0:  # on the floor
            face(-1, i, here - [0, 0, 0.5], 2, 1, joints.floor_pull * s_all[i])
        for axis in range(3):
            for sign in (-1, 1):
                other = (c[0] + sign * (axis == 2), c[1] + sign * (axis == 1), c[2] + sign * (axis == 0))
                centre = here.copy()
                centre[axis] += 0.5 * sign
                k = index.get(other)
                if k is not None:
                    if sign > 0:
                        face(i, k, centre, axis, sign, math.sqrt(s_all[i] * s_all[k]))
                elif other in fixed_cube:
                    face(-1, i, centre, axis, -sign, math.sqrt(s_all[i] * s_all[fixed_cube[other]]))
                elif other in terrain:
                    face(-1, i, centre, axis, -sign, joints.terrain_pull * s_all[i])
    if joints.edges:
        edge_force = 2 * joints.edge_share * joints.edge_pull
        for i, c in enumerate(cells):
            for ax_u in range(3):
                for ax_v in range(ax_u + 1, 3):
                    for su in (-1, 1):
                        for sv in (-1, 1):
                            step = np.zeros(3, dtype=int)
                            step[2 - ax_u] = su
                            step[2 - ax_v] = sv
                            other = tuple(int(t) for t in c + step)
                            k = index.get(other)
                            if k is not None and k < i:
                                continue
                            if k is None and other not in fixed_cube:
                                continue
                            u, v = su * AXES[ax_u], sv * AXES[ax_v]
                            mid = xyz[i] + 0.5 * u + 0.5 * v
                            # body a pushes body b away along u, v; a is the cube at i unless the other is fixed
                            a, b, uu, vv = (i, k, u, v) if k is not None else (-1, i, -u, -v)
                            j = len(pairs)
                            pairs.append((a, b, "edge"))
                            offs, dirs = _edge_template(uu, vv, joints.mu)
                            blocks.append((a, b, mid + offs, dirs, j))
                            s_ab = math.sqrt(s_all[i] * s_all[k if k is not None else fixed_cube[other]])
                            mag_a.append(a)
                            mag_b.append(b)
                            mag_p.append(mid)
                            mag_f.append(-edge_force * s_ab * (uu + vv) / math.sqrt(2))
    fixed = np.zeros((n, 6))
    if mag_a:
        ma, mb = np.array(mag_a), np.array(mag_b)
        mp, mf = np.array(mag_p), np.array(mag_f)
        for body, sign in ((mb, 1.0), (ma, -1.0)):
            live = body >= 0
            f = sign * mf[live]
            arm = mp[live] - xyz[body[live]]
            np.add.at(fixed, body[live], np.concatenate([f, np.cross(arm, f)], axis=1))
    if blocks:
        sizes = [len(bl[3]) for bl in blocks]
        A = np.repeat([bl[0] for bl in blocks], sizes)
        Bb = np.repeat([bl[1] for bl in blocks], sizes)
        J = np.repeat([bl[4] for bl in blocks], sizes)
        P = np.concatenate([bl[2] for bl in blocks])
        D = np.concatenate([bl[3] for bl in blocks])
    else:
        A = Bb = J = np.zeros(0, dtype=int)
        P = D = np.zeros((0, 3))
    return Interfaces(centres=xyz, a=A.astype(int), b=Bb.astype(int), p=P, d=D, joint=J.astype(int), pairs=pairs, fixed_force=fixed)


# -------------------------------------------------------------------------------------------------
# the LP: maximise lam s.t. sum_k x_k A_k + lam g = -f_fixed (- g_dead), lam + s = cap, x, lam, s >= 0


@dataclass
class Problem:
    """One LP in rows: row 0 the cap row, rows 1 + 6 i + (0..5) the equilibrium of body i."""

    rows: np.ndarray  # (K, 12) row index per generator entry, -1 for none
    vals: np.ndarray  # (K, 12)
    g: np.ndarray  # (m,) the load factor's column
    rhs: np.ndarray  # (m,)
    m: int
    scale: float = 1.0  # the equilibrium rows were divided by this


def rescaled(p, scale):
    """The same LP with its equilibrium rows divided by `scale` instead of p.scale."""
    g, rhs = p.g.copy(), p.rhs.copy()
    g[1:] *= p.scale / scale
    rhs[1:] *= p.scale / scale
    return Problem(rows=p.rows, vals=p.vals, g=g, rhs=rhs, m=p.m, scale=scale)


def problem(inter, load, dead=None, cap=50.0):
    """load: (n, 3) force per cube (at its centre) scaled by the factor; dead: (n, 3) unscaled."""
    n = len(inter.centres)
    m = 1 + 6 * n
    K = len(inter.a)
    rows = -np.ones((K, 12), dtype=int)
    vals = np.zeros((K, 12))
    for side, body, sign in ((0, inter.b, 1.0), (1, inter.a, -1.0)):
        live = body >= 0
        arm = inter.p - inter.centres[np.maximum(body, 0)]
        w = np.concatenate([inter.d, np.cross(arm, inter.d)], axis=1) * sign
        cols = slice(6 * side, 6 * side + 6)
        rows[live, cols] = 1 + 6 * body[live, None] + np.arange(6)
        vals[live, cols] = w[live]
    g = np.zeros(m)
    g[0] = 1.0
    g[1:] = np.concatenate([load, np.zeros((n, 3))], axis=1).reshape(-1)
    rhs = np.zeros(m)
    rhs[0] = cap
    rhs[1:] = -inter.fixed_force.reshape(-1)
    if dead is not None:
        rhs[1:] -= np.concatenate([dead, np.zeros((n, 3))], axis=1).reshape(-1)
    scale = max(1.0, float(np.abs(rhs[1:]).max(initial=0.0)), float(np.abs(g[1:]).max(initial=0.0)))  # forces O(1): the solver's tolerances are absolute
    g[1:] /= scale
    rhs[1:] /= scale
    return Problem(rows=rows, vals=vals, g=g, rhs=rhs, m=m, scale=scale)


def solve(problems, iterations=80, tol=1e-6, trace=False):  # trace: print residuals per iteration (debugging)
    """Batched Mehrotra IPM. Returns (factor (B,), duals list of (m,) arrays, converged (B,)). A
    structure that does not converge gets the largest factor among its primal-feasible iterates
    (equilibrium within tolerance with every force >= 0), which is a lower bound."""
    B = len(problems)
    if B == 0:
        return np.zeros(0), [], np.zeros(0, dtype=bool)
    dt = torch.float64
    m = max(p.m for p in problems)
    K = max(1, max(len(p.rows) for p in problems))
    rows = torch.full((B, K, 12), m, dtype=torch.long)  # m: a dummy row
    vals = torch.zeros((B, K, 12), dtype=dt)
    cmask = torch.zeros((B, K), dtype=dt)
    rmask = torch.zeros((B, m), dtype=dt)
    g = torch.zeros((B, m), dtype=dt)
    b = torch.zeros((B, m), dtype=dt)
    for i, p in enumerate(problems):
        k = len(p.rows)
        r = torch.as_tensor(p.rows)
        rows[i, :k] = torch.where(r >= 0, r, torch.full_like(r, m))
        vals[i, :k] = torch.as_tensor(p.vals)
        cmask[i, :k] = 1.0
        rmask[i, : p.m] = 1.0
        g[i, : p.m] = torch.as_tensor(p.g)
        b[i, : p.m] = torch.as_tensor(p.rhs)
    cap = b[:, 0].clone()
    nvar = cmask.sum(1) + 2
    sparse = m > 150  # large structures: A as (row, value) pairs and the normal matrix from 12 x 12 blocks (dense costs m^2 K)
    if sparse:
        flat = rows.reshape(B, -1)
        pair_index = (rows[:, :, :, None] * (m + 1) + rows[:, :, None, :]).reshape(B, -1)
        outer = (vals[:, :, :, None] * vals[:, :, None, :]).reshape(B, K, 144)
    else:
        dense = torch.zeros((B, m + 1, K), dtype=dt)  # A, with a dummy last row for absent bodies
        dense.scatter_add_(1, rows.permute(0, 2, 1), vals.permute(0, 2, 1))
        A = dense[:, :m, :] * cmask[:, None, :]
        del dense

    def A_mul(xs, xl, xsig):
        if sparse:
            out = torch.zeros((B, m + 1), dtype=dt)
            out.scatter_add_(1, flat, (vals * (xs * cmask)[:, :, None]).reshape(B, -1))
            out = out[:, :m]
        else:
            out = torch.bmm(A, xs[:, :, None])[:, :, 0]
        out = out + g * xl[:, None]
        out[:, 0] += xsig
        return out

    def AT_mul(y):
        if sparse:
            ypad = torch.cat([y, torch.zeros((B, 1), dtype=dt)], 1)
            ys = (torch.gather(ypad, 1, flat).reshape(B, K, 12) * vals).sum(-1) * cmask
        else:
            ys = torch.bmm(y[:, None, :], A)[:, 0, :]
        return ys, (g * y).sum(1), y[:, 0]

    def normal(ds, dl, dsig):
        if sparse:
            M = torch.zeros((B, (m + 1) * (m + 1)), dtype=dt)
            M.scatter_add_(1, pair_index, (outer * (ds * cmask)[:, :, None]).reshape(B, -1))
            M = M.reshape(B, m + 1, m + 1)[:, :m, :m]
        else:
            M = torch.bmm(A * ds[:, None, :], A.transpose(1, 2))
        M = M + dl[:, None, None] * g[:, :, None] * g[:, None, :]
        M[:, 0, 0] += dsig
        diag = torch.diagonal(M, dim1=1, dim2=2)
        scale = (diag * rmask).amax(1).clamp_min(1.0)
        return M + torch.diag_embed((1 - rmask) + REG * scale[:, None] * rmask)

    # variables: xs (B, K), xl, xsig (B,); duals y (B, m), zs, zl, zsig
    xs = torch.full((B, K), 10.0, dtype=dt) * cmask + (1 - cmask)
    xl = torch.ones(B, dtype=dt)
    xsig = (cap - 1.0).clamp_min(1.0)
    y = torch.zeros((B, m), dtype=dt)
    zs = torch.ones((B, K), dtype=dt)
    zl = torch.ones(B, dtype=dt)
    zsig = torch.ones(B, dtype=dt)
    done = torch.zeros(B, dtype=torch.bool)
    stalled = torch.zeros(B, dtype=torch.bool)
    best = torch.zeros(B, dtype=dt)  # the largest factor of a primal-feasible iterate: a certified lower bound
    bnorm = 1 + b.abs().amax(1)

    def ratio(x, dx, mask=None):
        neg = dx < 0
        if mask is not None:
            neg = neg & (mask > 0)
        r = torch.where(neg, -x / torch.where(neg, dx, -torch.ones_like(dx)), torch.full_like(x, math.inf))
        return r.amin(1) if r.dim() == 2 else r

    for _ in range(iterations):
        ax = A_mul(xs, xl, xsig)
        rp = (b - ax) * rmask
        ats, atl, atsig = AT_mul(y)
        rds = (0 - ats - zs) * cmask
        rdl = -1.0 - atl - zl
        rdsig = 0 - atsig - zsig
        mu = ((xs * zs * cmask).sum(1) + xl * zl + xsig * zsig) / nvar
        pres = rp.abs().amax(1) / bnorm
        dres = torch.maximum(rds.abs().amax(1), torch.maximum(rdl.abs(), rdsig.abs()))
        best = torch.where((pres < tol) & ~done & ~stalled, torch.maximum(best, xl), best)
        stalled = stalled | (~done & (mu < 1e-13 * (1 + xs.amax(1))))  # no progress left: keep the bound
        at_cap = (pres < tol) & (xl > cap * (1 - 1e-5))  # feasible at the cap: the answer is known
        done = done | at_cap | ((pres < tol) & (dres < tol) & (mu < 0.1 * tol * (1 + xs.amax(1))))  # sticky: a solved structure takes no more steps
        if bool((done | stalled).all()):
            break
        ds, dl, dsig = xs / zs * cmask, xl / zl, xsig / zsig
        M = normal(ds, dl, dsig)
        L, info = torch.linalg.cholesky_ex(M)
        if bool((info > 0).any()):
            M = M + torch.diag_embed(1e-8 * torch.diagonal(M, dim1=1, dim2=2).amax(1, keepdim=True).expand(-1, m))
            L, info = torch.linalg.cholesky_ex(M)

        def direction(rxs, rxl, rxsig):
            vs = (ds * rds - rxs / zs) * cmask
            vl = dl * rdl - rxl / zl
            vsig = dsig * rdsig - rxsig / zsig
            rhs = rp + A_mul(vs, vl, vsig) * rmask
            dy = torch.cholesky_solve(rhs[:, :, None], L)[:, :, 0] * rmask
            for _ in range(2):  # iterative refinement: the normal matrix is badly conditioned near the end
                dy = dy + torch.cholesky_solve((rhs - torch.bmm(M, dy[:, :, None])[:, :, 0])[:, :, None], L)[:, :, 0] * rmask
            ats_, atl_, atsig_ = AT_mul(dy)
            dzs = (rds - ats_) * cmask
            dzl = rdl - atl_
            dzsig = rdsig - atsig_
            dxs = (rxs - xs * dzs) / zs * cmask
            dxl = (rxl - xl * dzl) / zl
            dxsig = (rxsig - xsig * dzsig) / zsig
            return dxs, dxl, dxsig, dy, dzs, dzl, dzsig

        a = direction(-xs * zs * cmask, -xl * zl, -xsig * zsig)
        ap = torch.clamp(torch.minimum(torch.minimum(ratio(xs, a[0], cmask), ratio(xl, a[1])), ratio(xsig, a[2])), max=1.0)
        ad = torch.clamp(torch.minimum(torch.minimum(ratio(zs, a[4], cmask), ratio(zl, a[5])), ratio(zsig, a[6])), max=1.0)
        mu_aff = (((xs + ap[:, None] * a[0]) * (zs + ad[:, None] * a[4]) * cmask).sum(1) + (xl + ap * a[1]) * (zl + ad * a[5]) + (xsig + ap * a[2]) * (zsig + ad * a[6])) / nvar
        sigma = (mu_aff / mu).clamp(0, 1) ** 3
        sm = sigma * mu
        c = direction((-xs * zs - a[0] * a[4] + sm[:, None]) * cmask, -xl * zl - a[1] * a[5] + sm, -xsig * zsig - a[2] * a[6] + sm)
        ap = torch.clamp(0.99 * torch.minimum(torch.minimum(ratio(xs, c[0], cmask), ratio(xl, c[1])), ratio(xsig, c[2])), max=1.0)
        ad = torch.clamp(0.99 * torch.minimum(torch.minimum(ratio(zs, c[4], cmask), ratio(zl, c[5])), ratio(zsig, c[6])), max=1.0)
        if trace:
            print(f"pres {pres.max().item():.1e} dres {dres.max().item():.1e} mu {mu.max().item():.1e} ap {ap.min().item():.3f} ad {ad.min().item():.3f} lam {xl[0].item():.4f} best {best[0].item():.4f} xmax {xs.max().item():.1e} zmin {zs[cmask > 0].min().item():.1e}")
        keep = (~done & ~stalled).to(dt)
        ap, ad = ap * keep, ad * keep
        xs = xs + ap[:, None] * c[0]
        xl = xl + ap * c[1]
        xsig = xsig + ap * c[2]
        y = y + ad[:, None] * c[3]
        zs = zs + ad[:, None] * c[4] + (1 - cmask) * 0
        zl = zl + ad * c[5]
        zsig = zsig + ad * c[6]
    factor = torch.where(done, xl, best)  # unconverged: the best certified lower bound (conservative)
    return factor.numpy().copy(), [y[i, : p.m].numpy().copy() for i, p in enumerate(problems)], done.numpy().copy()


# -------------------------------------------------------------------------------------------------
# the check


@dataclass
class Result:
    factor: float  # the largest factor on the scaled loads (cap: at least that)
    stands: bool  # factor >= 1
    moving: np.ndarray  # (k, 3) grid cells of the cubes that move in the collapse (empty when it stands)
    opening: list = field(default_factory=list)  # [(cell, other cell or "fixed", kind)] joints that open or slide in that collapse
    converged: bool = True
    collapse: np.ndarray = None  # (k, 3) the cubes that move when the weakest component fails, at its factor (also when it stands)
    weak_joints: list = field(default_factory=list)  # the joints that open in that collapse


def components(cells, solid=(), anchored=(), edges=True):
    """Connected groups of free cubes (face or, with edges, edge neighbours); anchored and solid
    cells do not join groups."""
    cells = np.asarray(cells, dtype=int).reshape(-1, 3)
    index = {tuple(c): i for i, c in enumerate(cells)}
    parent = list(range(len(cells)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    steps = [s for s in np.ndindex(3, 3, 3) if 1 <= sum(v != 1 for v in s) <= (2 if edges else 1)]
    for i, c in enumerate(cells):
        for s in steps:
            k = index.get(tuple(c + np.array(s) - 1))
            if k is not None:
                parent[root(i)] = root(k)
    roots = np.array([root(i) for i in range(len(cells))], dtype=int)
    return [np.nonzero(roots == r)[0] for r in np.unique(roots)]


def _mechanism(inter, y, cells):
    """Cubes that move in the collapse (from the duals) and the joints between bodies that move
    differently."""
    n = len(inter.centres)
    v = y[1:].reshape(n, 6)
    speed = np.abs(v[:, :3]).max(1) + 0.5 * np.abs(v[:, 3:]).max(1)
    top = speed.max() if n else 0.0
    if top <= 1e-9:
        return np.zeros(n, dtype=bool), []
    moving = speed > 1e-3 * top
    opening = []
    vel = np.zeros((n + 1, 6))
    vel[:n] = v / top
    for j, (a, b, kind) in enumerate(inter.pairs):
        sel = inter.joint == j
        p = inter.p[sel].mean(0)
        va = vel[a] if a >= 0 else np.zeros(6)
        ua = va[:3] + np.cross(va[3:], p - (inter.centres[a] if a >= 0 else p))
        ub = vel[b][:3] + np.cross(vel[b][3:], p - inter.centres[b])
        if np.abs(ub - ua).max() > 1e-3:
            opening.append((tuple(int(t) for t in cells[b]), tuple(int(t) for t in cells[a]) if a >= 0 else "fixed", kind))
    return moving, opening


def check_many(structures, joints=Joints(), batch=16):
    """structures: list of dicts with keys cells, and optionally solid, anchored, load ((n, 3)
    physical force per cube in module weights, or {cell: (fx, fy, fz)}), strength, mode
    ("gravity": the factor scales gravity and the load together; "live": gravity stays at 1 g and
    the factor scales the load), weights ((n,) each cube's weight in module weights, default 1),
    mu (friction for this structure). Returns a Result per structure."""
    jobs = []  # (structure index, component cell indices, Interfaces, Problem)
    results = [None] * len(structures)
    for s, spec in enumerate(structures):
        cells = np.asarray(spec["cells"], dtype=int).reshape(-1, 3)
        n = len(cells)
        load = _load_array(spec.get("load"), cells)
        mode = spec.get("mode", "gravity")
        strength = spec.get("strength")
        anchored = np.asarray(spec.get("anchored", ()), dtype=int).reshape(-1, 3)
        weights = np.ones(n) if spec.get("weights") is None else np.asarray(spec["weights"], dtype=float)[:n]
        own = replace(joints, mu=spec["mu"]) if spec.get("mu") is not None else joints
        for comp in components(cells, spec.get("solid", ()), anchored, joints.edges):
            st = None if strength is None else np.concatenate([np.asarray(strength)[comp], np.asarray(strength)[n:] if len(strength) > n else np.ones(len(anchored))])
            inter = interfaces(cells[comp], spec.get("solid", ()), anchored, st, own)
            weight = np.zeros((len(comp), 3))
            weight[:, 2] = -weights[comp]
            if mode == "gravity":
                prob = problem(inter, weight + load[comp], cap=joints.cap)
            else:
                prob = problem(inter, load[comp], dead=weight, cap=joints.cap)
            jobs.append((s, comp, inter, prob, mode, weight))
    # live mode is feasible at factor 0 only if gravity alone holds: check gravity first
    live_jobs = [k for k, job in enumerate(jobs) if job[4] == "live"]
    gravity_ok = {}
    if live_jobs:
        probe = [problem(jobs[k][2], jobs[k][5], cap=1.0) for k in live_jobs]
        f, _, _ = _solve_chunked(probe, batch)
        gravity_ok = {k: f[i] >= 1 - EPS for i, k in enumerate(live_jobs)}
    run = [k for k in range(len(jobs)) if gravity_ok.get(k, True)]
    factor, duals, conv = _solve_chunked([jobs[k][3] for k in run], batch)
    out = {k: (factor[i], duals[i], conv[i]) for i, k in enumerate(run)}
    per = {}
    for k, (s, comp, inter, prob, mode, _) in enumerate(jobs):
        cells = np.asarray(structures[s]["cells"], dtype=int).reshape(-1, 3)
        if k in out:
            f, y, ok = out[k]
            moving, opening = _mechanism(inter, y, cells[comp]) if f < joints.cap - EPS else (np.zeros(len(comp), dtype=bool), [])
        else:
            f, ok = 0.0, True  # fails under its own weight before any live load
            moving, opening = np.ones(len(comp), dtype=bool), []
        per.setdefault(s, []).append((float(min(f, joints.cap)), cells[comp][moving], opening, bool(ok)))
    empty = np.zeros((0, 3), dtype=int)
    for s in range(len(structures)):
        parts = per.get(s, [])
        if not parts:
            results[s] = Result(factor=joints.cap, stands=True, moving=empty)
            continue
        weakest = min(parts, key=lambda p: p[0])
        failing = [p for p in parts if p[0] < 1 - EPS]
        results[s] = Result(
            factor=weakest[0],
            stands=weakest[0] >= 1 - EPS,
            moving=np.concatenate([p[1] for p in failing]) if failing else empty,
            opening=[o for p in failing for o in p[2]],
            converged=all(p[3] for p in parts),
            collapse=weakest[1],
            weak_joints=weakest[2],
        )
    stuck = [s for s, r in enumerate(results) if not r.converged]
    if stuck and joints.edges:  # without edge contacts the LP converges more often, and it is a lower bound (fewer joints)
        retry = check_many([structures[s] for s in stuck], replace(joints, edges=False), batch)
        for s, r in zip(stuck, retry):
            if r.factor > results[s].factor:
                r.converged = False  # still only a lower bound for the structure with its edge contacts
                results[s] = r
    return results


def _solve_chunked(problems, batch):
    """Solve in batches of similar size (padding to a batch's largest costs its cube). Structures
    that do not converge with forces scaled to O(1) are tried again unscaled (the IPM fails on
    different ones each way); otherwise the larger certified lower bound is kept."""
    factor, duals, conv = _solve_sizes(problems, batch)
    again = [i for i in range(len(problems)) if not conv[i] and problems[i].scale != 1.0]
    if again:
        f2, d2, c2 = _solve_sizes([rescaled(problems[i], 1.0) for i in again], batch)
        for j, i in enumerate(again):
            if c2[j]:
                factor[i], duals[i], conv[i] = f2[j], d2[j] * problems[i].scale, True
            elif f2[j] > factor[i]:
                factor[i] = f2[j]
    return factor, duals, conv


def _solve_sizes(problems, batch):
    order = sorted(range(len(problems)), key=lambda i: problems[i].m)
    factor = np.zeros(len(problems))
    duals = [None] * len(problems)
    conv = np.zeros(len(problems), dtype=bool)
    chunks, chunk = [], []
    for i in order:
        size = (len(chunk) + 1) * problems[i].m * max(len(p.rows) for p in [problems[i]] + [problems[k] for k in chunk])
        if chunk and (len(chunk) >= batch or problems[i].m > 1.25 * problems[chunk[0]].m + 12 or size > 4e7):
            chunks.append(chunk)
            chunk = []
        chunk.append(i)
    if chunk:
        chunks.append(chunk)
    for chunk in chunks:
        f, y, c = solve([problems[i] for i in chunk])
        for j, i in enumerate(chunk):
            factor[i], duals[i], conv[i] = f[j], y[j], c[j]
    return factor, duals, conv


def _load_array(load, cells):
    out = np.zeros((len(cells), 3))
    if load is None:
        return out
    if isinstance(load, dict):
        index = {tuple(c): i for i, c in enumerate(cells)}
        for cell, f in load.items():
            out[index[tuple(cell)]] += f
        return out
    return np.asarray(load, dtype=float).reshape(-1, 3)


def check(cells, solid=(), anchored=(), load=None, mode="gravity", strength=None, weights=None, mu=None, joints=Joints()):
    """One structure (check_many's keys as arguments)."""
    return check_many([{"cells": cells, "solid": solid, "anchored": anchored, "load": load, "mode": mode, "strength": strength, "weights": weights, "mu": mu}], joints)[0]


def check_grid(occupied, solid=None, extra=None, joints=Joints()):
    """A voxel scene: occupied (D, H, W) bool cubes, solid (D, H, W) bool terrain (walls, objects,
    held fixed), extra (D, H, W) float downward load resting on each cube in module weights (an
    object's weight). Returns a Result; Result.moving are the cubes that would fall."""
    cells = np.argwhere(np.asarray(occupied))
    terrain = np.argwhere(np.asarray(solid)) if solid is not None else np.zeros((0, 3), dtype=int)
    load = None
    if extra is not None:
        load = np.zeros((len(cells), 3))
        load[:, 2] = -np.asarray(extra)[tuple(cells.T)]
    return check(cells, solid=terrain, load=load, joints=joints)
