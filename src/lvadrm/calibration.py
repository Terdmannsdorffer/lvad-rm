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


# ------------------------------------------------------------------------------------------
# Heart-failure patient under LVAD support (HANDOFF section 7)

# Fernandez-Zapico et al. (2024), Table 1: baseline HF patient at rest, LVAD at 2300-2500 rpm.
# Mapping to our outputs: TCO -> co, QLV -> qlv, wedge pressure -> mean LAP, AOP -> mean AoP,
# AOPES / AOPED -> max / min AoP, LVES -> min LVV, LVED -> LVV at the end of the beat.
HF_TARGETS = {
    "co": (4.9, 5.1),
    "qlv": (0.7, 1.2),
    "lap_mean": (14.0, 16.0),
    "aop_mean": (82.0, 86.0),
    "aop_max": (98.0, 100.0),
    "aop_min": (68.0, 75.0),
    "lvv_min": (129.0, 131.0),
    "edv": (179.0, 183.0),
}
# priority (HANDOFF 7): output, aortic pressure, LV volumes and aortic valve flow
HF_WEIGHTS = {"co": 1.0, "qlv": 1.0, "aop_mean": 1.0, "lvv_min": 1.0, "edv": 1.0, "lap_mean": 0.5, "aop_max": 0.5, "aop_min": 0.5}
HF_HR = 80.0
HF_RS_FZ = 0.9  # SVR of Fernandez-Zapico Table 2 (their return node is the CVP, ours the left atrium)

HF_FIT_NAMES = ("emax", "emin", "v0", "vtot", "omega", "rs")
HF_BOUNDS = {
    "emax": (0.2, 3.0),
    "emin": (0.02, 0.4),
    "v0": (0.0, 100.0),
    "vtot": (200.0, 1200.0),
    "omega": (5000.0, 16000.0),
    "rs": (0.4, 1.6),
}


def hf_metrics(stats_last) -> dict[str, np.ndarray]:
    return {k: np.asarray(getattr(stats_last, k)) for k in HF_TARGETS}


def hf_residuals(metrics: dict[str, np.ndarray]) -> np.ndarray:
    """Weighted relative errors to the middle of each target range, shape [..., n_targets]."""
    res = []
    for k, (lo, hi) in HF_TARGETS.items():
        mid = 0.5 * (lo + hi)
        res.append(HF_WEIGHTS[k] * (metrics[k] - mid) / mid)
    return np.stack(res, axis=-1)


def _hf_batch(x: np.ndarray, names, fixed: dict, n_beats: int, sim_kwargs: dict):
    """Steady-state last-beat stats for a batch of parameter vectors x [B, len(names)]."""
    vals = {n: jnp.asarray(x[:, i]) for i, n in enumerate(names)} | {k: v for k, v in fixed.items() if k not in names}
    vtot = vals.pop("vtot")
    omega = vals.pop("omega")
    p = make_params(hr=HF_HR, **vals)
    _, stats, _ = run_beats(p, vtot, omega, n_beats=n_beats, **sim_kwargs)
    return jax.tree.map(lambda a: np.asarray(a)[..., -1], stats)


def calibrate_hf_patient(
    fit_rs: bool = True, n_global: int = 600, n_local: int = 3, n_beats: int = 25, max_nfev: int = 60,
    seed: int = 0, **sim_kwargs
):
    """Least-squares fit of Emax, Emin, V0, Vtot, omega (and Rs) to Fernandez-Zapico Table 1.

    Global random search (``n_global`` candidates simulated in one batch), then
    ``scipy.optimize.least_squares`` from the ``n_local`` best candidates with batched
    finite-difference Jacobians.
    Returns a dict with the fitted parameters, metrics, residuals and per-target range check.
    """
    from scipy.optimize import least_squares

    names = HF_FIT_NAMES if fit_rs else HF_FIT_NAMES[:-1]
    fixed = {"rs": HF_RS_FZ}
    lo = np.array([HF_BOUNDS[n][0] for n in names])
    hi = np.array([HF_BOUNDS[n][1] for n in names])
    scale = hi - lo

    def fun(x):
        s = _hf_batch(x[None], names, fixed, n_beats, sim_kwargs)
        return hf_residuals(hf_metrics(s))[0]

    def jac(x):
        h = 1e-4 * scale
        xs = np.concatenate([x[None], x[None] + np.diag(h)])
        r = hf_residuals(hf_metrics(_hf_batch(xs, names, fixed, n_beats, sim_kwargs)))
        return ((r[1:] - r[0]) / h[:, None]).T

    rng = np.random.default_rng(seed)
    cand = lo + rng.uniform(0.0, 1.0, (n_global, len(names))) * scale
    r = hf_residuals(hf_metrics(_hf_batch(cand, names, fixed, n_beats, sim_kwargs)))
    cost = np.where(np.all(np.isfinite(r), axis=-1), np.sum(r**2, axis=-1), np.inf)
    best = None
    for i in np.argsort(cost)[:n_local]:
        sol = least_squares(fun, cand[i], jac=jac, bounds=(lo, hi), x_scale=scale, xtol=1e-8, ftol=1e-10, max_nfev=max_nfev)
        if best is None or sol.cost < best.cost:
            best = sol
    s = _hf_batch(best.x[None], names, fixed, n_beats, sim_kwargs)
    m = {k: float(v[0]) for k, v in hf_metrics(s).items()}
    params = {n: float(v) for n, v in zip(names, best.x)} | ({} if fit_rs else {"rs": HF_RS_FZ})
    return {
        "params": params,
        "hr": HF_HR,
        "metrics": m,
        "in_range": {k: bool(lo_ <= m[k] <= hi_) for k, (lo_, hi_) in HF_TARGETS.items()},
        "cost": float(best.cost),
        "esv_at_avc": float(s.esv[0]),
        "qp_min": float(s.qp_min[0]),
        "lvp_min": float(s.lvp_min[0]),
    }
