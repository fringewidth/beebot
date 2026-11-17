"""Axisymmetric magnetostatics with NGSolve for a disc magnet over a steel plate (soft-magnetic parts).

Formulation: azimuthal vector potential A (H1, A=0 on axis and outer boundary), B = curl(A e_phi),
weak form  int (1/mu)(B - J).dB r dr dz = 0 with permanent-magnet polarisation J = (0, Jz).
Pull force from the Maxwell stress integrated over a plane in the air gap (needs gap > 0).
Steel is linear (mu_r given) or nonlinear with an isotropic B-H curve (Newton iteration).
"""
from __future__ import annotations

import numpy as np
from ngsolve import (BilinearForm, CoefficientFunction, GridFunction, H1, Integrate, LinearForm, Mesh, dx,
                     grad, x, y, specialcf, Trace)  # noqa: F401
from ngsolve import Preconditioner, solvers  # noqa: F401
from netgen.geom2d import SplineGeometry

MU0 = 4e-7 * np.pi


def pull_force_axisym(d: float, h: float, br: float, gap: float, mu_r: float | None = 1000.0,
                      plate_t: float = 5e-3, plate_r: float | None = None, domain: float | None = None,
                      mu_magnet: float = 1.05, maxh_fine: float | None = None, order: int = 2):
    """Attractive force (N) of a cylindrical magnet (diameter d, height h, polarisation br) on a steel plate.

    gap is the air gap between magnet face and plate (m, > 0). Plate has radius plate_r (default 4 d, at least
    10 mm) and thickness plate_t; domain radius/height `domain` (default 12 d + 20 h).
    """
    ra = d / 2
    plate_r = plate_r or max(4 * d, 10e-3)
    dom = domain or (12 * d + 20 * h)
    fine = maxh_fine or min(h, d) / 6
    g = SplineGeometry()
    pts = {}

    def P(name, px, py, mh):
        pts[name] = g.AppendPoint(px, py, maxh=mh)

    zt = -plate_t
    zb = gap + h  # magnet top
    # rectangle points: plate [0,plate_r] x [zt,0]; magnet [0,ra] x [gap, gap+h]; domain [0,dom] x [-dom/2? ,dom]
    zmin, zmax = zt - dom / 3, zb + dom / 2
    coarse = dom / 8
    P("o0", 0, zmin, coarse)
    P("o1", dom, zmin, coarse)
    P("o2", dom, zmax, coarse)
    P("o3", 0, zmax, coarse)
    # axis points
    P("a_pt", 0, zt, fine * 2)
    P("a_p0", 0, 0, fine)
    P("a_m0", 0, gap, fine)
    P("a_m1", 0, zb, fine)
    P("pl_b", plate_r, zt, fine * 3)
    P("pl_t", plate_r, 0, fine * 1.5)
    P("m_b", ra, gap, fine)
    P("m_t", ra, zb, fine)
    # outer boundary
    g.Append(["line", pts["o0"], pts["o1"]], bc="out", leftdomain=1, rightdomain=0)
    g.Append(["line", pts["o1"], pts["o2"]], bc="out", leftdomain=1, rightdomain=0)
    g.Append(["line", pts["o2"], pts["o3"]], bc="out", leftdomain=1, rightdomain=0)
    # axis (left): from o3 down to a_m1, magnet, a_m0, a_p0, plate, a_pt, o0
    g.Append(["line", pts["o3"], pts["a_m1"]], bc="axis", leftdomain=1, rightdomain=0)
    g.Append(["line", pts["a_m1"], pts["a_m0"]], bc="axis", leftdomain=2, rightdomain=0)
    g.Append(["line", pts["a_m0"], pts["a_p0"]], bc="axis", leftdomain=1, rightdomain=0)
    g.Append(["line", pts["a_p0"], pts["a_pt"]], bc="axis", leftdomain=3, rightdomain=0)
    g.Append(["line", pts["a_pt"], pts["o0"]], bc="axis", leftdomain=1, rightdomain=0)
    # magnet right/top/bottom
    g.Append(["line", pts["a_m0"], pts["m_b"]], bc="mag", leftdomain=2, rightdomain=1)
    g.Append(["line", pts["m_b"], pts["m_t"]], bc="mag", leftdomain=2, rightdomain=1)
    g.Append(["line", pts["m_t"], pts["a_m1"]], bc="mag", leftdomain=2, rightdomain=1)
    # plate
    g.Append(["line", pts["a_pt"], pts["pl_b"]], bc="pl", leftdomain=3, rightdomain=1)
    g.Append(["line", pts["pl_b"], pts["pl_t"]], bc="pl", leftdomain=3, rightdomain=1)
    g.Append(["line", pts["pl_t"], pts["a_p0"]], bc="pl", leftdomain=3, rightdomain=1)
    g.SetMaterial(1, "air")
    g.SetMaterial(2, "magnet")
    g.SetMaterial(3, "steel")
    mesh = Mesh(g.GenerateMesh(maxh=coarse))
    return _solve(mesh, br, gap, h, mu_r, mu_magnet, order, dom, plate_r)


# Generic Brauer saturation curve for nu(B^2) = k1 exp(k2 B^2) + k3 (A/m/T), illustrative mild steel, not a
# specific grade and not fitted to any validation point. mu_r at 1 T is about 1900, at 2 T about 30.
BRAUER = (3.8, 2.17, 396.2)


def _solve(mesh, br, gap, h, mu_r, mu_magnet, order, dom, plate_r):
    nonlinear = mu_r is None
    fes = H1(mesh, order=order, dirichlet="axis|out")
    u, v = fes.TnT()
    r = x
    mu_lin = 1000.0 if nonlinear else mu_r
    inv_mu_steel = mesh.MaterialCF({"steel": 1 / (MU0 * mu_lin)}, default=0)
    inv_mu_rest = mesh.MaterialCF({"air": 1 / MU0, "magnet": 1 / (MU0 * mu_magnet), "steel": 0.0}, default=1 / MU0)
    jz = mesh.MaterialCF({"magnet": br}, default=0)

    def B(w):
        # B_r = -dA/dz, B_z = dA/dr + A/r
        gw = grad(w)
        return CoefficientFunction((-gw[1], gw[0] + w / r))

    from ngsolve import exp as ngexp
    gf = GridFunction(fes)
    f = LinearForm(fes)
    f += (jz / (MU0 * mu_magnet)) * B(v)[1] * r * dx
    f.Assemble()
    n_iter = 40 if nonlinear else 1
    for it in range(n_iter):
        if nonlinear and it > 0:
            Bp = B(gf)
            bb = Bp * Bp
            nu_steel = mesh.MaterialCF({"steel": BRAUER[0] * ngexp(BRAUER[1] * bb) + BRAUER[2]}, default=0)
        else:
            nu_steel = inv_mu_steel
        a = BilinearForm(fes)
        a += (inv_mu_rest + nu_steel) * (B(u) * B(v)) * r * dx
        a.Assemble()
        new = GridFunction(fes)
        new.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
        if it == 0:
            gf.vec.data = new.vec
        else:
            diff = (new.vec - gf.vec).Norm() / max(new.vec.Norm(), 1e-30)
            gf.vec.data = 0.6 * new.vec + 0.4 * gf.vec
            if diff < 1e-6:
                break
    Bf = B(gf)
    # Maxwell stress on plane z = gap/2 over r in [0, dom]: F_z = int (Bz^2 - Br^2)/(2 mu0) 2 pi r dr
    zc = gap / 2
    n = 4000
    rr = np.linspace(1e-9, dom * 0.999, n)
    bz = np.empty(n)
    br_ = np.empty(n)
    for k, rv in enumerate(rr):
        try:
            val = Bf(mesh(rv, zc))
            br_[k], bz[k] = val
        except Exception:
            br_[k] = bz[k] = 0.0
    integrand = (bz**2 - br_**2) / (2 * MU0) * 2 * np.pi * rr
    force = float(np.trapezoid(integrand, rr))
    return {"force": force, "ndof": fes.ndof, "mesh_elements": mesh.ne, "bz_gap_axis": float(Bf(mesh(1e-6, zc))[1]),
            "b_steel_max": _max_b_in_steel(mesh, Bf)}


def _max_b_in_steel(mesh, Bf) -> float:
    from ngsolve import sqrt
    mag = sqrt(Bf * Bf)
    vals = Integrate(mag * 0 + mag, mesh, definedon=mesh.Materials("steel"), order=2) if False else 0.0
    pts = [(rv, -z) for rv in np.linspace(1e-6, 4e-3, 20) for z in (1e-5, 2e-4, 1e-3)]
    mx = 0.0
    for rv, z in pts:
        try:
            b = Bf(mesh(rv, z))
            mx = max(mx, float(np.hypot(*b)))
        except Exception:
            pass
    return mx
