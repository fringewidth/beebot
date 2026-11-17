"""Top-level toy objectives for demonstrating opt.py (picklable under spawn)."""
import numpy as np

from .magnetics import pull_force_magnet_magnet


def zdt1(x):
    x = np.asarray(x)
    g = 1 + 9 * np.mean(x[1:])
    return [x[0], g * (1 - np.sqrt(x[0] / g))]


def sphere(x):
    return float(np.sum((np.asarray(x) - 0.3) ** 2))


def toy_latch(params):
    """Toy success test: two 1 mm magnets seated with a tolerance-affected gap must pull at least 0.05 N.

    Uses magnet position error (lateral, mm-scale) and magnetisation scale from the tolerance helper.
    """
    lat = float(np.hypot(params["magnet_pos_x"], params["magnet_pos_y"]))
    br = 1.45 * float(params["magnetisation"])
    gap = float(params["print_dim_gap"])
    f = pull_force_magnet_magnet("cyl", (1e-3, 1e-3), br, gap, lateral=lat)
    # also must hold a sliding load by stiction: friction * pull >= load
    return bool(f >= params["threshold"] and params["friction_mu"] * f >= params["load"])


def toy_latch_objectives(x):
    """x = (gap_mm, diameter_mm): minimise gap-penalty and maximise pull per volume -> two objectives."""
    gap, d = x[0] * 1e-3, x[1] * 1e-3
    f = pull_force_magnet_magnet("cyl", (d, d), 1.45, gap)
    vol = np.pi * (d / 2) ** 2 * d
    return [-f, vol * 1e9]
