"""Fixed-step RK4 integration, one cardiac beat at a time, with per-beat statistics.

A beat is split into ``n_seg`` segments of ``n_inner`` RK4 substeps (N_sub = n_seg * n_inner,
dt = tc / N_sub). Statistics are accumulated on the fly; selected signals are sampled at the
end of each segment (e.g. n_seg = 64 gives a 64-sample pump-flow waveform per beat, and
n_inner = 1 gives the full trajectory for plotting).

The pump speed inside a beat is omega(t) = omega0 + omega_rate * t (t beat-local), so a
zero-order hold per beat is omega_rate = 0 and the open-loop ramps of Simaan Fig. 12 are exact.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from lvadrm.elastance import cycle_length
from lvadrm.model import AOP, G, LVV, QP, rhs, total_volume
from lvadrm.params import Params

SAMPLE_NAMES = ("qp", "lvp", "lvv", "aop", "ap", "lap", "qa")


class BeatStats(NamedTuple):
    """Per-beat outputs (HANDOFF section 5.7). Flows in ml/s unless noted."""

    tc: jax.Array  # beat length [s]
    omega_mean: jax.Array  # mean pump speed [rpm]
    qp_min: jax.Array
    qp_max: jax.Array
    qp_mean: jax.Array
    dqp_min: jax.Array  # min dQp/dt [ml/s^2]
    dqp_max: jax.Array
    edv: jax.Array  # LVV at the end of the beat [ml]
    edp: jax.Array  # LVP at the end of the beat [mmHg]
    esv: jax.Array  # LVV at aortic valve closure (NaN if the valve did not open) [ml]
    esp: jax.Array  # LVP at aortic valve closure [mmHg]
    lvv_min: jax.Array
    lvv_max: jax.Array
    lvp_min: jax.Array
    lvp_max: jax.Array
    aop_min: jax.Array
    aop_max: jax.Array
    aop_mean: jax.Array
    ap_mean: jax.Array  # mean systemic arterial pressure x3 [mmHg]
    lap_mean: jax.Array
    co: jax.Array  # cardiac output = mean x5 [l/min]
    qlv: jax.Array  # mean flow through the aortic valve [l/min]
    v_aortic: jax.Array  # volume ejected through the aortic valve, integral of Qa [ml]
    suction_frac: jax.Array  # fraction of the beat with x1 <= x1_bar
    vtot: jax.Array  # total volume at the end of the beat [ml]


def rk4_step(f, t, y, dt):
    """One RK4 step for (dy, g) = f(t, y). Returns (y_new, integral of g over the step, k1, g1)."""
    k1, g1 = f(t, y)
    k2, g2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3, g3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4, g4 = f(t + dt, y + dt * k3)
    y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    g_int = (dt / 6.0) * (g1 + 2.0 * g2 + 2.0 * g3 + g4)
    return y_new, g_int, k1, g1


def _samples(y, g):
    return jnp.stack([g[G["qp"]], g[G["lvp"]], y[LVV], g[G["aop"]], g[G["ap"]], g[G["lap"]], g[G["qa"]]])


@partial(jax.jit, static_argnames=("n_seg", "n_inner", "pump_on", "pump_form", "unroll"))
def simulate_beat(
    y0,
    p: Params,
    omega0,
    omega_rate=0.0,
    *,
    n_seg: int = 64,
    n_inner: int = 160,
    pump_on: bool = True,
    pump_form: str = "consistent",
    unroll: int = 1,
):
    """Integrate one beat from beat-local t = 0 to tc. Single environment (use ``jax.vmap``).

    Returns (y_end, BeatStats, samples[n_seg, len(SAMPLE_NAMES)]).
    """
    n_sub = n_seg * n_inner
    tc = cycle_length(p.heart.hr)
    dt = tc / n_sub
    dtype = y0.dtype

    def f(t, y):
        return rhs(t, y, omega0 + omega_rate * t, p, pump_on, pump_form)

    # running extrema of [qp, dqp, lvp, lvv, aop]
    inf = jnp.asarray(jnp.inf, dtype)
    nan = jnp.asarray(jnp.nan, dtype)
    carry0 = (
        y0,
        jnp.zeros(len(G), dtype),
        jnp.full(5, inf),
        jnp.full(5, -inf),
        jnp.asarray(False),
        nan,
        nan,
    )

    def substep(carry, k):
        y, ints, mn, mx, open_prev, lvv_avc, lvp_avc = carry
        t = k * dt
        y_new, g_int, k1, g1 = rk4_step(f, t, y, dt)
        vals = jnp.stack([g1[G["qp"]], k1[QP], g1[G["lvp"]], y[LVV], g1[G["aop"]]])
        open_now = g1[G["qa"]] > 0.0
        closing = open_prev & ~open_now
        lvv_avc = jnp.where(closing, y[LVV], lvv_avc)
        lvp_avc = jnp.where(closing, g1[G["lvp"]], lvp_avc)
        carry = (y_new, ints + g_int, jnp.minimum(mn, vals), jnp.maximum(mx, vals), open_now, lvv_avc, lvp_avc)
        return carry, None

    def segment(carry, s):
        ks = s * n_inner + jnp.arange(n_inner)
        carry, _ = jax.lax.scan(substep, carry, ks, unroll=unroll)
        y = carry[0]
        t_end = (s + 1) * n_inner * dt
        _, g = f(t_end, y)
        return carry, _samples(y, g)

    carry, samples = jax.lax.scan(segment, carry0, jnp.arange(n_seg))
    y_end, ints, mn, mx, _, lvv_avc, lvp_avc = carry
    _, g_end = f(tc, y_end)

    stats = BeatStats(
        tc=tc,
        omega_mean=omega0 + 0.5 * omega_rate * tc,
        qp_min=mn[0],
        qp_max=mx[0],
        qp_mean=ints[G["qp"]] / tc,
        dqp_min=mn[1],
        dqp_max=mx[1],
        edv=y_end[LVV],
        edp=g_end[G["lvp"]],
        esv=lvv_avc,
        esp=lvp_avc,
        lvv_min=mn[3],
        lvv_max=mx[3],
        lvp_min=mn[2],
        lvp_max=mx[2],
        aop_min=mn[4],
        aop_max=mx[4],
        aop_mean=ints[G["aop"]] / tc,
        ap_mean=ints[G["ap"]] / tc,
        lap_mean=ints[G["lap"]] / tc,
        co=ints[G["qt"]] / tc * 0.06,
        qlv=ints[G["qa"]] / tc * 0.06,
        v_aortic=ints[G["qa"]],
        suction_frac=ints[G["suction"]] / tc,
        vtot=total_volume(y_end, p),
    )
    stats = jax.tree.map(lambda x: jnp.asarray(x, dtype), stats)
    return y_end, stats, samples


@partial(jax.jit, static_argnames=("n_seg", "n_inner", "pump_on", "pump_form", "unroll"))
def simulate_beats(
    y0,
    p: Params,
    omega0s,
    omega_rates=None,
    p_seq: Params | None = None,
    *,
    n_seg: int = 64,
    n_inner: int = 160,
    pump_on: bool = True,
    pump_form: str = "consistent",
    unroll: int = 1,
):
    """Integrate consecutive beats (one per entry of ``omega0s``). Single environment.

    ``p_seq``, if given, is a Params pytree with a leading beat axis that overrides ``p``
    beat by beat (parameter changes are applied at beat boundaries, tn = 0).
    Returns (y_end, BeatStats with a leading beat axis, samples[n_beats, n_seg, n_samples]).
    """
    if omega_rates is None:
        omega_rates = jnp.zeros_like(omega0s)

    def body(y, xs):
        w0, wr, pb = xs
        pb = p if pb is None else pb
        y, stats, samples = simulate_beat(
            y, pb, w0, wr, n_seg=n_seg, n_inner=n_inner, pump_on=pump_on, pump_form=pump_form, unroll=unroll
        )
        return y, (stats, samples)

    y_end, (stats, samples) = jax.lax.scan(body, y0, (omega0s, omega_rates, p_seq))
    return y_end, stats, samples
