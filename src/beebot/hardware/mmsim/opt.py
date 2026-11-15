"""Parallel multi-objective search (pymoo NSGA-II, GA; cma) and Monte Carlo tolerance analysis.

Objectives must be top-level (picklable) callables f(x: ndarray) -> float or sequence of floats, all minimised.
At most MAX_WORKERS processes are used.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

MAX_WORKERS = 6


def _cap(workers: int) -> int:
    return max(1, min(int(workers), MAX_WORKERS))


def _eval_one(args):
    fn, x = args
    return np.atleast_1d(np.asarray(fn(x), float))


@dataclass
class SearchResult:
    algo: str
    X: np.ndarray  # decision vectors of the final non-dominated set (or best for single objective)
    F: np.ndarray  # objectives of those
    n_evals: int
    history_best: list


def run_search(fn, lower, upper, n_obj: int = 1, algo: str = "nsga2", pop: int = 24, n_gen: int = 20,
               workers: int = 4, seed: int = 0, ctx: str = "spawn") -> SearchResult:
    """Minimise fn over the box [lower, upper]. algo in {'nsga2', 'ga', 'cma'}; evaluations run in parallel."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    workers = _cap(workers)
    pool = mp.get_context(ctx).Pool(workers)
    try:
        if algo == "cma":
            return _run_cma(fn, lower, upper, pop, n_gen, seed, pool)
        return _run_pymoo(fn, lower, upper, n_obj, algo, pop, n_gen, seed, pool)
    finally:
        pool.close()
        pool.join()


def _run_pymoo(fn, lower, upper, n_obj, algo, pop, n_gen, seed, pool):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    class P(Problem):
        def __init__(self):
            super().__init__(n_var=len(lower), n_obj=n_obj, xl=lower, xu=upper)

        def _evaluate(self, X, out, *a, **k):
            res = pool.map(_eval_one, [(fn, x) for x in X])
            out["F"] = np.vstack(res)

    alg = NSGA2(pop_size=pop) if algo == "nsga2" else GA(pop_size=pop)
    hist = []

    def cb(algorithm):
        hist.append(float(np.min(algorithm.pop.get("F")[:, 0])))

    from pymoo.core.callback import Callback

    class CB(Callback):
        def notify(self, algorithm):
            cb(algorithm)

    res = minimize(P(), alg, ("n_gen", n_gen), seed=seed, verbose=False, callback=CB())
    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    return SearchResult(algo, X, F, int(res.algorithm.evaluator.n_eval), hist)


def _run_cma(fn, lower, upper, pop, n_gen, seed, pool):
    import cma

    x0 = (lower + upper) / 2
    sigma0 = 0.25
    scale = upper - lower
    es = cma.CMAEvolutionStrategy(
        (x0 - lower) / scale, sigma0,
        {"popsize": pop, "bounds": [0, 1], "seed": seed + 1, "verbose": -9},
    )
    hist, n = [], 0
    best_x, best_f = None, math.inf
    for _ in range(n_gen):
        xs = es.ask()
        real = [lower + np.asarray(z) * scale for z in xs]
        fs = [float(r[0]) for r in pool.map(_eval_one, [(fn, x) for x in real])]
        n += len(fs)
        es.tell(xs, fs)
        k = int(np.argmin(fs))
        if fs[k] < best_f:
            best_f, best_x = fs[k], real[k]
        hist.append(best_f)
    return SearchResult("cma", np.atleast_2d(best_x), np.atleast_2d([best_f]), n, hist)


# ---------------------------------------------------------------- Monte Carlo tolerances

@dataclass(frozen=True)
class Tolerance:
    """Perturbation of one parameter. kind 'abs' (half-width in parameter units) or 'rel' (fraction).

    dist 'uniform' draws within +-half_width; 'normal' uses half_width as two standard deviations.
    """

    kind: str
    half_width: float
    dist: str = "uniform"

    def sample(self, rng, nominal, size=None):
        scale = self.half_width * (abs(nominal) if self.kind == "rel" else 1.0)
        if self.dist == "normal":
            return nominal + rng.normal(0, scale / 2, size)
        return nominal + rng.uniform(-scale, scale, size)


# Tolerances from the project brief: magnet position +-0.05 mm, magnetisation +-5 %, friction +-30 %,
# print dimensions +-0.05 mm. Keys are parameter-name prefixes.
DEFAULT_TOLERANCES = {
    "magnet_pos": Tolerance("abs", 0.05e-3),
    "magnetisation": Tolerance("rel", 0.05),
    "friction": Tolerance("rel", 0.30),
    "print_dim": Tolerance("abs", 0.05e-3),
}


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def perturb(params: dict, tolerances: dict, rng) -> dict:
    out = dict(params)
    for key, val in params.items():
        for prefix, tol in tolerances.items():
            if key.startswith(prefix):
                arr = np.asarray(val, float)
                out[key] = tol.sample(rng, arr, arr.shape if arr.shape else None)
                break
    return out


def _mc_chunk(args):
    fn, params, tolerances, seed, n = args
    rng = np.random.default_rng(seed)
    k = 0
    for _ in range(n):
        if fn(perturb(params, tolerances, rng)):
            k += 1
    return k


def monte_carlo(fn, params: dict, n: int = 200, tolerances: dict | None = None, workers: int = 4,
                seed: int = 0, ctx: str = "spawn") -> dict:
    """Success rate of boolean fn(params_dict) under random tolerance perturbation, with Wilson 95 % interval.

    fn must be a top-level callable. Seeds are derived per chunk, so results are reproducible for fixed
    (n, workers, seed).
    """
    tolerances = tolerances or DEFAULT_TOLERANCES
    workers = _cap(workers)
    sizes = [n // workers + (1 if i < n % workers else 0) for i in range(workers)]
    jobs = [(fn, params, tolerances, seed * 1000 + i, s) for i, s in enumerate(sizes) if s > 0]
    with mp.get_context(ctx).Pool(len(jobs)) as pool:
        ks = pool.map(_mc_chunk, jobs)
    k = int(sum(ks))
    lo, hi = wilson_interval(k, n)
    return {"n": n, "successes": k, "rate": k / n, "wilson_lo": lo, "wilson_hi": hi}


def format_mc(res: dict) -> str:
    return "success %d/%d = %.3f, Wilson 95%% [%.3f, %.3f]" % (
        res["successes"], res["n"], res["rate"], res["wilson_lo"], res["wilson_hi"])
