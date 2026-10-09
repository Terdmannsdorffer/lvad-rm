"""High-accuracy reference integration with SciPy (float64), beat by beat.

Uses the same right-hand side as the JAX integrator (``lvadrm.model.rhs``), so a comparison
measures integration error only. The per-beat integrals of ``g`` are carried as extra
quadrature states. Requires ``jax.config.update("jax_enable_x64", True)``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from lvadrm.elastance import cycle_length
from lvadrm.integrate import SAMPLE_NAMES
from lvadrm.model import G, G_NAMES, LVV, QP, STATE_NAMES, rhs, rhs_pressure_form, total_volume
from lvadrm.params import Params

N_STATE = len(STATE_NAMES)


def _check_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError("reference integration needs jax_enable_x64=True")


@partial(jax.jit, static_argnames=("pump_on", "pump_form", "pressure_form"))
def _aug_rhs(t, z, omega0, omega_rate, p, pump_on, pump_form, pressure_form):
    f = rhs_pressure_form if pressure_form else rhs
    dy, g = f(t, z[:N_STATE], omega0 + omega_rate * t, p, pump_on, pump_form)
    return jnp.concatenate([dy, g])


@partial(jax.jit, static_argnames=("pump_on", "pump_form", "pressure_form"))
def _aug_jac(t, z, omega0, omega_rate, p, pump_on, pump_form, pressure_form):
    return jax.jacfwd(lambda zz: _aug_rhs(t, zz, omega0, omega_rate, p, pump_on, pump_form, pressure_form))(z)


@partial(jax.jit, static_argnames=("pump_on", "pump_form", "pressure_form"))
def _g_batch(ts, ys, omega0, omega_rate, p, pump_on, pump_form, pressure_form):
    f = rhs_pressure_form if pressure_form else rhs
    return jax.vmap(lambda t, y: f(t, y, omega0 + omega_rate * t, p, pump_on, pump_form)[1])(ts, ys)


def reference_beat(
    y0,
    p: Params,
    omega0: float = 0.0,
    omega_rate: float = 0.0,
    *,
    pump_on: bool = True,
    pump_form: str = "consistent",
    pressure_form: bool = False,
    n_eval: int = 20001,
    method: str = "Radau",
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_step: float = 1e-3,
):
    """Integrate one beat (beat-local t from 0 to tc) with ``solve_ivp``.

    Returns a dict with ``t`` [n_eval], ``y`` [n_eval, 6], ``g`` [n_eval, len(G_NAMES)],
    ``ints`` (integrals of g over the beat) and ``y_end``. With ``pressure_form`` the first
    state is LVP instead of LVV.
    """
    _check_x64()
    tc = float(cycle_length(p.heart.hr))
    n_y = len(y0)
    z0 = np.concatenate([np.asarray(y0, dtype=np.float64), np.zeros(len(G_NAMES))])
    args = (jnp.float64(omega0), jnp.float64(omega_rate), p)
    static = {"pump_on": pump_on, "pump_form": pump_form, "pressure_form": pressure_form}

    def fun(t, z):
        return np.asarray(_aug_rhs(t, z, *args, **static))

    def jac(t, z):
        return np.asarray(_aug_jac(t, z, *args, **static))

    t_eval = np.linspace(0.0, tc, n_eval)
    sol = solve_ivp(
        fun,
        (0.0, tc),
        z0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        jac=jac if method in ("Radau", "BDF", "LSODA") else None,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    ys = sol.y[:n_y].T
    g = np.asarray(_g_batch(jnp.asarray(sol.t), jnp.asarray(ys), *args, **static))
    return {
        "t": sol.t,
        "y": ys,
        "g": g,
        "ints": sol.y[n_y:, -1],
        "y_end": sol.y[:n_y, -1],
        "nfev": sol.nfev,
    }


def stats_from_trajectory(traj: dict, p: Params, omega0: float = 0.0, omega_rate: float = 0.0) -> dict:
    """Per-beat statistics from a dense trajectory (same definitions as ``integrate.BeatStats``)."""
    t, y, g, ints = traj["t"], traj["y"], traj["g"], traj["ints"]
    tc = t[-1]
    # extrema over [0, tc) like the on-the-fly accumulator
    gi, yi = g[:-1], y[:-1]
    qa_open = gi[:, G["qa"]] > 0.0
    closing = np.nonzero(qa_open[:-1] & ~qa_open[1:])[0]
    if len(closing):
        k = closing[-1] + 1
        esv, esp = yi[k, LVV], gi[k, G["lvp"]]
    else:
        esv = esp = np.nan
    dqp = np.gradient(y[:, QP], t)[:-1]
    return {
        "tc": tc,
        "omega_mean": omega0 + 0.5 * omega_rate * tc,
        "qp_min": gi[:, G["qp"]].min(),
        "qp_max": gi[:, G["qp"]].max(),
        "qp_mean": ints[G["qp"]] / tc,
        "t_qp_min": t[:-1][np.argmin(gi[:, G["qp"]])],
        "v_back": ints[G["qp_back"]],
        "dqp_min": dqp.min(),
        "dqp_max": dqp.max(),
        "edv": y[-1, LVV],
        "edp": g[-1, G["lvp"]],
        "esv": esv,
        "esp": esp,
        "lvv_min": yi[:, LVV].min(),
        "lvv_max": yi[:, LVV].max(),
        "lvp_min": gi[:, G["lvp"]].min(),
        "lvp_max": gi[:, G["lvp"]].max(),
        "lvp_mean": ints[G["lvp"]] / tc,
        "aop_min": gi[:, G["aop"]].min(),
        "aop_max": gi[:, G["aop"]].max(),
        "aop_mean": ints[G["aop"]] / tc,
        "ap_mean": ints[G["ap"]] / tc,
        "lap_mean": ints[G["lap"]] / tc,
        "co": ints[G["qt"]] / tc * 0.06,
        "qlv": ints[G["qa"]] / tc * 0.06,
        "v_aortic": ints[G["qa"]],
        "suction_frac": ints[G["suction"]] / tc,
        "vtot": float(total_volume(np.asarray(y[-1]), p)),
    }


def reference_samples(traj: dict) -> np.ndarray:
    """Signals in ``integrate.SAMPLE_NAMES`` order from a dense trajectory, shape [n_eval, n]."""
    y, g = traj["y"], traj["g"]
    cols = {"lvv": y[:, LVV]}
    cols.update({name: g[:, G[name]] for name in SAMPLE_NAMES if name != "lvv"})
    return np.stack([cols[name] for name in SAMPLE_NAMES], axis=-1)
