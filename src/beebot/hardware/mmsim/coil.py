"""Planar spiral coil on flex: geometry, resistance, inductance, pulse heating.

Inductance has four independent routes: Mohan current-sheet, modified Wheeler, the Aebischer 2019
correction of the Jenei formula (square spirals, 2..20 turns), and a magpylib flux-linkage integral.
Thermal model: copper node and polyimide node with temperature-dependent resistance.
"""
from __future__ import annotations

from dataclasses import dataclass

import magpylib as mp
import numpy as np
from scipy.integrate import solve_ivp

MU0 = 4e-7 * np.pi

# Material constants (SI). Copper resistivity at 20 C: 1.678e-8 (Aebischer 2019 uses this value);
# IACS annealed copper is 1.724e-8. Temperature coefficient 0.00393 /K.
RHO_CU_20 = 1.678e-8
ALPHA_CU = 0.00393
DENSITY_CU = 8960.0
CP_CU = 385.0
# Polyimide (Kapton HN class), from memory of the DuPont datasheet, not re-verified here.
DENSITY_PI = 1420.0
CP_PI = 1090.0
K_PI = 0.12


@dataclass(frozen=True)
class Spiral:
    """Planar spiral. outer is the mid-conductor outer side length (square) or diameter (round).

    trace, space and cu_t in metres. layers are connected in series with the same winding sense;
    layer_pitch is the centre to centre spacing between copper layers (flex dielectric).
    """

    turns: int
    outer: float
    trace: float
    space: float
    cu_t: float = 18e-6
    layers: int = 1
    shape: str = "square"
    layer_pitch: float = 25e-6

    @property
    def pitch(self) -> float:
        return self.trace + self.space

    @property
    def turns_total(self) -> int:
        return self.turns * self.layers

    def inner(self) -> float:
        return self.outer - 2 * (self.turns - 1) * self.pitch


def spiral_vertices(sp: Spiral, z: float = 0.0, flip: bool = False) -> np.ndarray:
    """Polyline of one layer, centred on the origin, plane z. Square follows Aebischer fig. 1."""
    w, A, N = sp.pitch, sp.outer, sp.turns
    if sp.shape == "round":
        # Archimedean spiral, mid-conductor radius from A/2 inward, 90 points per turn
        n = 90 * N
        th = np.linspace(0, 2 * np.pi * N, n + 1)
        r = A / 2 - w * th / (2 * np.pi)
        pts = np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n + 1, z)])
        if flip:
            pts[:, 1] *= -1
        return pts
    pts = [(0.0, A / 2)]
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    x, y = pts[0]
    lens = [A / 2] + [A - w * ((k - 1) // 2) for k in range(1, 4 * N)]
    # final side: half of the innermost side as in the paper's convention
    lens.append((A - w * ((4 * N - 1) // 2)) / 2)
    for k, ln in enumerate(lens):
        dx, dy = dirs[k % 4]
        x, y = x + dx * ln, y + dy * ln
        pts.append((x, y))
    arr = np.array([(px, py, z) for px, py in pts])
    if flip:
        arr[:, 1] *= -1
    return arr


def stack_vertices(sp: Spiral) -> np.ndarray:
    """All layers joined in series (alternate layers traversed outward, so winding sense is preserved)."""
    pieces = []
    for l in range(sp.layers):
        v = spiral_vertices(sp, z=l * sp.layer_pitch)
        if l % 2 == 1:
            v = v[::-1]
            # reversed traversal flips current sense; mirror y so the sense matches layer 0
            v = v.copy()
            v[:, 1] *= -1
        pieces.append(v)
    return np.vstack(pieces)


def polyline_length(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(v, axis=0), axis=1).sum())


def track_length(sp: Spiral) -> float:
    """Total conductor length over all layers (metres)."""
    return polyline_length(spiral_vertices(sp)) * sp.layers


def resistivity(T: float = 20.0) -> float:
    return RHO_CU_20 * (1 + ALPHA_CU * (T - 20.0))


def resistance(sp: Spiral, T: float = 20.0) -> float:
    return resistivity(T) * track_length(sp) / (sp.trace * sp.cu_t)


# ---------------------------------------------------------------- inductance formulas

def _dims(sp: Spiral):
    dout = sp.outer + sp.trace
    din = dout - 2 * (sp.turns * sp.trace + (sp.turns - 1) * sp.space)
    return dout, max(din, 1e-9)


def inductance_mohan(sp: Spiral) -> float:
    """Mohan et al. 1999 current-sheet expression, one layer (square or circular)."""
    c = (1.27, 2.07, 0.18, 0.13) if sp.shape == "square" else (1.00, 2.46, 0.00, 0.20)
    dout, din = _dims(sp)
    dav = (dout + din) / 2
    phi = (dout - din) / (dout + din)
    return MU0 * sp.turns**2 * dav * c[0] / 2 * (np.log(c[1] / phi) + c[2] * phi + c[3] * phi**2)


def inductance_wheeler(sp: Spiral) -> float:
    """Modified Wheeler expression (Mohan 1999), one layer."""
    k1, k2 = (2.34, 2.75) if sp.shape == "square" else (2.25, 3.55)
    dout, din = _dims(sp)
    dav = (dout + din) / 2
    rho = (dout - din) / (dout + din)
    return k1 * MU0 * sp.turns**2 * dav / (1 + k2 * rho)


def inductance_aebischer(sp: Spiral) -> float:
    """Aebischer 2019 formula (eq. 25), square spirals, 2 to 20 turns. Transcribed from the paper's code."""
    N, A, w, s, h = sp.turns, sp.outer, sp.pitch, sp.trace, sp.cu_t
    mu2 = MU0 / (2 * np.pi)
    rho = ((N - 1) * w + s) / (A - (N - 1) * w)
    Ai = A - 2 * (N - 1) * w
    l = 4 * N * Ai + (4 * N**2 - 5 * N) * w
    dp4 = 4 * N * (N + 1) / 3 * w
    if N == 2:
        c1 = c2 = 0.0
    elif N <= 7:
        c1 = -(7.2 * rho + 0.35) / (2.78 * rho + 1)
        c2 = -12.8 * rho**2 + 11 * rho + 0.80
    elif N <= 12:
        c1 = -(2.1 * rho + 0.17) / (0.75 * rho + 1)
        c2 = -(0.59 * rho + 1.2) / (-0.90 * rho + 1)
    else:
        c1 = -(1.5 * rho + 0.11) / (0.98 * rho + 1)
        c2 = -(2.85 * rho + 2.48) / (-0.75 * rho + 1)
    lp = mu2 * l * (np.log(1 / N / (s + h) * l) - 0.19315)
    mm = mu2 * l * 0.46716
    mp_ = mu2 * l * (np.log(np.sqrt(1 + (l / dp4) ** 2) + l / dp4) - np.sqrt(1 + (dp4 / l) ** 2) + dp4 / l)
    L0 = lp - N * mm + (N - 1) * mp_
    return L0 * (1 - (c1 * N + c2) / 100)


# ---------------------------------------------------------------- flux-linkage inductance (magpylib)

def _contour(shape: str, half: float, n_side: int):
    """Gauss-Legendre nodes along a closed contour: points C(t), tangent-weighted arc nodes.

    Returns (pts (m,2), cross (m,) = weights times (C x C') so that area element = s * cross ds).
    """
    x, wgt = np.polynomial.legendre.leggauss(n_side)
    x = (x + 1) / 2
    wgt = wgt / 2
    if shape == "square":
        corners = np.array([[half, -half], [half, half], [-half, half], [-half, -half]])
        P, W = [], []
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            for xi, wi in zip(x, wgt):
                p = a + (b - a) * xi
                d = b - a
                P.append(p)
                W.append(wi * (p[0] * d[1] - p[1] * d[0]))
        return np.array(P), np.array(W)
    n = 4 * n_side
    t = (np.arange(n) + 0.5) / n * 2 * np.pi
    P = half * np.column_stack([np.cos(t), np.sin(t)])
    W = np.full(n, half * half * 2 * np.pi / n)
    return P, W


def _loop_obj(shape: str, half: float, z: float):
    if shape == "square":
        v = np.array([[half, -half, z], [half, half, z], [-half, half, z], [-half, -half, z], [half, -half, z]])
        return mp.current.Polyline(current=1.0, vertices=v)
    return mp.current.Circle(current=1.0, diameter=2 * half, position=(0, 0, z))


def flux_through(shape: str, src_half: float, src_z: float, area_half: float, area_z: float,
                 breaks=(), n_side: int = 12, n_rad: int = 24) -> float:
    """Flux (Wb) of a unit-current loop through the surface bounded by a concentric contour (same shape)."""
    src = _loop_obj(shape, src_half, src_z)
    P, W = _contour(shape, area_half, n_side)
    edges = np.array(sorted(set([0.0, 1.0] + [b for b in breaks if 0 < b < 1])))
    xg, wg = np.polynomial.legendre.leggauss(n_rad)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        u = (xg + 1) / 2
        # cos clustering toward both ends
        sgrid = a + (b - a) * (1 - np.cos(np.pi * u)) / 2
        dsdu = (b - a) * np.pi * np.sin(np.pi * u) / 2
        wr = wg / 2 * dsdu
        S = sgrid[:, None, None] * P[None, :, :]
        pts = np.concatenate([S.reshape(-1, 2), np.full((S.shape[0] * S.shape[1], 1), area_z)], axis=1)
        bz = src.getB(pts)[:, 2].reshape(len(sgrid), len(P))
        total += float(np.sum(bz * sgrid[:, None] * W[None, :] * wr[:, None]))
    return total


def ring_sides(sp: Spiral):
    """Half-sides (or radii) of the equivalent concentric rings, matching the spiral's total length."""
    mean_len = polyline_length(spiral_vertices(sp)) / sp.turns
    k = np.arange(sp.turns)
    if sp.shape == "square":
        mean_side = mean_len / 4
        sides = mean_side - 2 * sp.pitch * (k - (sp.turns - 1) / 2)
        return sides / 2
    mean_r = mean_len / (2 * np.pi)
    return mean_r - sp.pitch * (k - (sp.turns - 1) / 2)


def inductance_flux(sp: Spiral, n_side: int = 12, n_rad: int = 24) -> float:
    """Inductance from magpylib flux linkage of the equivalent nested rings.

    Self terms use Maxwell's geometric mean distance of the conductor cross-section (0.2235 (w + t)) as an
    inward offset of the surface boundary. Mutual terms integrate the outer ring's field through the inner
    ring's surface (smooth integrand). Concentric rings approximate the true spiral.
    """
    halves = ring_sides(sp)
    gmd = 0.2235 * (sp.trace + sp.cu_t)
    loops = [(h, l * sp.layer_pitch) for l in range(sp.layers) for h in halves]
    L = 0.0
    for i, (hi, zi) in enumerate(loops):
        for j, (hj, zj) in enumerate(loops):
            if i == j:
                L += flux_through(sp.shape, hi, zi, hi - gmd, zi, n_side=n_side, n_rad=n_rad)
            elif hj >= hi:
                # loop j (outer or equal) field through area of i
                L += flux_through(sp.shape, hj, zj, hi, zi, n_side=n_side, n_rad=n_rad)
            else:
                L += flux_through(sp.shape, hi, zi, hj, zj, n_side=n_side, n_rad=n_rad)
    return L


# ---------------------------------------------------------------- electrical operating point

def ampere_turns(sp: Spiral, v_supply: float, r_series: float = 0.0, T: float = 20.0):
    """Coil current and ampere-turns for a DC supply across coil plus a series resistance."""
    r = resistance(sp, T)
    i = v_supply / (r + r_series)
    return {"R": r, "I": i, "NI": i * sp.turns_total, "P_coil": i * i * r}


# ---------------------------------------------------------------- thermal

def _thermal_params(sp: Spiral, pi_thickness: float, h_conv: float, adiabatic: bool):
    area_trace = sp.trace * track_length(sp) / sp.layers  # contact area per layer (one face)
    v_cu = sp.trace * sp.cu_t * track_length(sp)
    c_cu = DENSITY_CU * CP_CU * v_cu
    footprint = (sp.outer + sp.trace) ** 2 if sp.shape == "square" else np.pi * ((sp.outer + sp.trace) / 2) ** 2
    # polyimide heated volume: footprint area times thickness (both sides share one layer count)
    v_pi = footprint * pi_thickness * (sp.layers + 1)
    c_pi = DENSITY_PI * CP_PI * v_pi
    g_cp = K_PI * area_trace * 2 * sp.layers / max(pi_thickness / 2, 1e-12)
    g_amb = 0.0 if adiabatic else h_conv * footprint * 2
    return c_cu, c_pi, g_cp, g_amb


def pulse_temperature(sp: Spiral, current: float, t_pulse: float, t_amb: float = 25.0,
                      pi_thickness: float = 25e-6, h_conv: float = 10.0, include_pi: bool = True,
                      adiabatic: bool = False, n_eval: int = 400):
    """Copper and polyimide temperature (C) during a constant-current pulse.

    Two lumped nodes. include_pi False gives a copper-only node; adiabatic True also removes convection
    and gives the Onderdonk limit. h_conv in W/m2/K (natural convection in air is 5 to 25).
    """
    c_cu, c_pi, g_cp, g_amb = _thermal_params(sp, pi_thickness, h_conv, adiabatic)
    r20 = resistance(sp, 20.0)
    if not include_pi:
        g_cp = 0.0

    def rhs(t, y):
        tc, tp = y
        p = current**2 * r20 * (1 + ALPHA_CU * (tc - 20.0))
        q = g_cp * (tc - tp)
        dtc = (p - q) / c_cu
        dtp = (q - g_amb * (tp - t_amb)) / c_pi if include_pi else 0.0
        if not include_pi:
            dtc = (p - g_amb * (tc - t_amb)) / c_cu
        return [dtc, dtp]

    def too_hot(t, y):
        return y[0] - 1500.0

    too_hot.terminal = True
    ts = np.linspace(0, t_pulse, n_eval)
    sol = solve_ivp(rhs, (0, t_pulse), [t_amb, t_amb], t_eval=ts, rtol=1e-8, atol=1e-8, method="LSODA",
                    events=too_hot)
    if sol.status == 1:
        ys = np.column_stack([sol.y[0], sol.y[1]]) if sol.y.size else np.array([[1500.0, t_amb]])
        return np.append(sol.t, sol.t_events[0]), np.append(sol.y[0], 1500.0), np.append(sol.y[1], sol.y[1][-1] if sol.y.size else t_amb)
    return sol.t, sol.y[0], sol.y[1]


def peak_temperature(sp, current, t_pulse, **kw) -> float:
    return float(pulse_temperature(sp, current, t_pulse, **kw)[1].max())


def safe_current(sp: Spiral, t_pulse: float, t_max: float = 100.0, t_amb: float = 25.0, **kw) -> float:
    """Largest constant current (A) whose copper peak temperature stays at or below t_max (C)."""
    lo, hi = 0.0, 1.0
    while peak_temperature(sp, hi, t_pulse, t_amb=t_amb, **kw) < t_max:
        hi *= 2
        if hi > 1e4:
            return np.inf
    for _ in range(40):
        mid = (lo + hi) / 2
        if peak_temperature(sp, mid, t_pulse, t_amb=t_amb, **kw) < t_max:
            lo = mid
        else:
            hi = mid
    return lo


def safe_current_table(sp: Spiral, durations, t_max: float = 100.0, **kw):
    return [(float(t), safe_current(sp, t, t_max, **kw)) for t in durations]


def onderdonk_current(area_m2: float, t: float, t_melt: float = 1083.0, t_amb: float = 20.0) -> float:
    """Onderdonk fusing current (A) for copper of cross-section area_m2 and duration t (s)."""
    cmil = area_m2 / (np.pi / 4 * (1e-3 * 25.4e-3) ** 2)
    return cmil * np.sqrt(np.log10((t_melt - t_amb) / (234.0 + t_amb) + 1) / (33.0 * t))
