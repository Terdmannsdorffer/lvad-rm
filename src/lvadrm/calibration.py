"""Steady-state helpers and calibration of the total blood volume (HANDOFF section 5.5).

The model is a closed loop, so the total volume Vtot (conserved exactly) sets the operating
point. Simaan et al. (2009) do not report it; we fit it to their healthy unassisted heart
(Fig. 5: HR 75, 117/77 mmHg, MAP 99 mmHg, CO 5.21 l/min, SV 69.5 ml). Pressures are taken
at the aortic node x4 (AoP), which is what Fig. 5 plots.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lvadrm.integrate import simulate_beats
from lvadrm.model import initial_state
from lvadrm.params import Params, make_params

HEALTHY_TARGETS = {"sbp": 117.0, "dbp": 77.0, "map": 99.0, "co": 5.21, "sv": 69.5}


def batch_axes(tree):
    """``vmap`` in_axes for a pytree: 0 for leaves with ndim >= 1, None for scalars."""
    return jax.tree.map(lambda a: 0 if jnp.ndim(a) > 0 else None, tree)


def run_beats(p: Params, vtot, omega=0.0, *, n_beats: int = 40, **sim_kwargs):
    """Simulate ``n_beats`` at constant speed from the default initial guess with total volume ``vtot``.

    Any leaf of ``p``, ``vtot`` or ``omega`` may be a 1-D array (one value per environment);
    all batched leaves must share the same length. Returns (y_end, stats[..., n_beats], samples).
    """

    def one(pp, vt, w):
        y0 = initial_state(pp, vt)
        return simulate_beats(y0, pp, jnp.full(n_beats, w, dtype=y0.dtype), **sim_kwargs)

    axes = (batch_axes(p), batch_axes(vtot), batch_axes(omega))
    if not jax.tree.leaves(axes):  # nothing batched (None leaves are dropped)
        return one(p, vtot, omega)
    return jax.vmap(one, in_axes=axes)(p, vtot, omega)


def healthy_metrics(stats) -> dict[str, np.ndarray]:
    """Fig. 5 quantities from the last beat of unassisted runs (stats with a trailing beat axis)."""
    last = jax.tree.map(lambda x: np.asarray(x)[..., -1], stats)
    return {
        "sbp": last.aop_max,
        "dbp": last.aop_min,
        "map": last.aop_mean,
        "co": last.co,
        "sv": last.v_aortic,
    }


def _cost(metrics: dict[str, np.ndarray]) -> np.ndarray:
    return sum(((metrics[k] - v) / v) ** 2 for k, v in HEALTHY_TARGETS.items())


def calibrate_healthy_vtot(
    emin: float = 0.05,
    emax: float = 2.0,
    v0: float = 10.0,
    hr: float = 75.0,
    rs: float = 1.0,
    bounds: tuple[float, float] = (240.0, 340.0),
    n_grid: int = 41,
    n_rounds: int = 3,
    n_beats: int = 40,
    **sim_kwargs,
) -> tuple[float, dict[str, float]]:
    """Least-squares fit of Vtot (relative errors on SBP, DBP, MAP, CO, SV) by nested grid search."""
    lo, hi = bounds
    best = None
    for _ in range(n_rounds):
        vtots = np.linspace(lo, hi, n_grid)
        p = make_params(emax=emax, emin=emin, v0=v0, hr=hr, rs=rs)
        _, stats, _ = run_beats(p, jnp.asarray(vtots), n_beats=n_beats, pump_on=False, **sim_kwargs)
        metrics = healthy_metrics(stats)
        cost = _cost(metrics)
        i = int(np.argmin(cost))
        best = (float(vtots[i]), {k: float(v[i]) for k, v in metrics.items()} | {"cost": float(cost[i])})
        step = vtots[1] - vtots[0]
        lo, hi = vtots[i] - step, vtots[i] + step
    return best
