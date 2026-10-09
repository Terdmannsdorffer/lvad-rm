"""Steady-state speed thresholds (operating window) at constant pump speed.

For each condition (a set of model parameters and a total volume) and each monotone predicate
of the periodic-regime beat statistics (False below the threshold speed, True above), find the
threshold speed with a coarse grid followed by bisection.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from lvadrm.calibration import run_beats
from lvadrm.labels import EPS_AVO_ML
from lvadrm.params import make_params


def default_predicates(back_ml: float = 0.0, eps_avo: float = EPS_AVO_ML, x1_bar: float = 1.0) -> dict[str, Callable]:
    return {
        "omega_back": lambda s: s.v_back <= back_ml,  # no more backflow through the pump
        "omega_avc": lambda s: s.v_aortic <= eps_avo,  # aortic valve no longer opens
        "omega_suc": lambda s: s.lvp_min <= x1_bar,  # suction resistance switches on
    }


def steady_last(cond: dict[str, np.ndarray], vtot: np.ndarray, omega: np.ndarray, n_beats: int, sim_kwargs: dict):
    """Last-beat statistics at constant speed for a batch of conditions (all arrays of equal length)."""
    p = make_params(**{k: jnp.asarray(v) for k, v in cond.items()})
    _, stats, _ = run_beats(p, jnp.asarray(vtot), jnp.asarray(omega), n_beats=n_beats, **sim_kwargs)
    return jax.tree.map(lambda x: np.asarray(x)[..., -1], stats)


def thresholds(
    cond: dict[str, np.ndarray],
    vtot: np.ndarray,
    grid: np.ndarray,
    predicates: dict[str, Callable],
    n_bisect: int = 9,
    n_beats: int = 30,
    **sim_kwargs,
):
    """Threshold speeds per condition. Returns (dict name -> [n_c] array (NaN if not bracketed or
    not monotone on the grid), grid statistics with shape [n_c, n_grid])."""
    n_c, n_g = len(vtot), len(grid)
    rep = {k: np.repeat(np.asarray(v), n_g) for k, v in cond.items()}
    s = steady_last(rep, np.repeat(vtot, n_g), np.tile(grid, n_c), n_beats, sim_kwargs)
    grid_stats = jax.tree.map(lambda x: x.reshape(n_c, n_g), s)

    out = {}
    for name, pred in predicates.items():
        flags = pred(grid_stats)
        lo = np.full(n_c, np.nan)
        hi = np.full(n_c, np.nan)
        for c in range(n_c):
            idx = np.nonzero(flags[c])[0]
            if len(idx) and idx[0] > 0 and np.all(flags[c, idx[0]:]):
                lo[c], hi[c] = grid[idx[0] - 1], grid[idx[0]]
        ok = ~np.isnan(lo)
        if ok.any():
            sub = {k: np.asarray(v)[ok] for k, v in cond.items()}
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                f = pred(steady_last(sub, vtot[ok], mid[ok], n_beats, sim_kwargs))
                lo[ok] = np.where(f, lo[ok], mid[ok])
                hi[ok] = np.where(f, mid[ok], hi[ok])
        out[name] = np.where(ok, 0.5 * (lo + hi), np.nan)
    return out, grid_stats
