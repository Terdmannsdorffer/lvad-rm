"""Patient sampling and within-episode events (HANDOFF section 8).

A scenario is a base value for each time-varying parameter (Rs, HR, Emax, Vtot) plus up to
K events. Event j moves parameter p_j linearly from its current value to a target over
[start_j, start_j + duration_j] seconds (10 s for HR, 50 s for the others, as Fernandez-Zapico
et al. 2024 following Ursino 1998). ``schedule`` turns a scenario into per-beat parameter
values; parameters change at beat boundaries (tn = 0) and time advances by 60 / HR per beat.

Vtot changes are applied by the environment through the left-atrial volume source q_vol
(an assumption: the model has no venous compartment, so preload drops are expressed as a
change of the total volume of the loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

VARYING = ("rs", "hr", "emax", "vtot")
RS, HR, EMAX, VTOT = range(len(VARYING))
TRANSITION_S = (50.0, 10.0, 50.0, 50.0)


@dataclass(frozen=True)
class Patient:
    """Base patient: fixed parameters plus the base values of the time-varying ones."""

    emax: float
    emin: float
    v0: float
    vtot: float
    rs: float
    hr: float
    omega: float  # pump speed of the calibrated operating point [rpm]


@dataclass(frozen=True)
class ScenarioConfig:
    """Sampling ranges. HR is absolute; Rs, Emax and Vtot are multipliers of the base patient.

    Defaults: the "maximum range" of Fernandez-Zapico et al. 2024 (Table 2) relative to their
    defaults (SVR 0.4-1.4 of 0.9, Els 0.4-1.2 of 0.8), HR up to Ferreira's strenuous exercise (135),
    and a Vtot change of -15 % / +10 % (assumption; their TBV range is 5300-5800 of 5800 ml).
    """

    rs_scale: tuple[float, float] = (0.4 / 0.9, 1.4 / 0.9)
    hr_range: tuple[float, float] = (60.0, 135.0)
    hr0_range: tuple[float, float] = (60.0, 100.0)  # initial (resting) heart rate
    emax_scale: tuple[float, float] = (0.5, 1.5)
    vtot_scale: tuple[float, float] = (0.85, 1.10)
    n_events_max: int = 3


# The initial HANDOFF ranges (Rs 0.5-2.0 absolute, i.e. relative to Simaan's Rs = 1): wider and
# hypertension-heavy for the HF patient (Rs base 0.79). Kept for generalization experiments.
WIDE_SCENARIOS = ScenarioConfig(rs_scale=(0.5 / 0.79298, 2.0 / 0.79298))


class Scenario(NamedTuple):
    base: jax.Array  # [4] initial values of (rs, hr, emax, vtot)
    ev_param: jax.Array  # [K] int
    ev_start: jax.Array  # [K] s
    ev_dur: jax.Array  # [K] s
    ev_target: jax.Array  # [K]
    ev_active: jax.Array  # [K] bool


def _ranges(patient: Patient, cfg: ScenarioConfig) -> np.ndarray:
    return np.array(
        [
            (cfg.rs_scale[0] * patient.rs, cfg.rs_scale[1] * patient.rs),
            cfg.hr_range,
            (cfg.emax_scale[0] * patient.emax, cfg.emax_scale[1] * patient.emax),
            (cfg.vtot_scale[0] * patient.vtot, cfg.vtot_scale[1] * patient.vtot),
        ]
    )


def sample_scenario(key, patient: Patient, cfg: ScenarioConfig, episode_beats: int) -> Scenario:
    """Random patient (initial values) and random events spread over the episode."""
    k_base, k_hr, k_n, k_p, k_s, k_t = jax.random.split(key, 6)
    rng = jnp.asarray(_ranges(patient, cfg), dtype=jnp.float32)
    base = jax.random.uniform(k_base, (4,), minval=rng[:, 0], maxval=rng[:, 1])
    base = base.at[HR].set(jax.random.uniform(k_hr, (), minval=cfg.hr0_range[0], maxval=cfg.hr0_range[1]))
    k = cfg.n_events_max
    n_events = jax.random.randint(k_n, (), 0, k + 1)
    ev_param = jax.random.randint(k_p, (k,), 0, 4)
    duration_est = episode_beats * 60.0 / base[HR]
    ev_start = jax.random.uniform(k_s, (k,), minval=0.0, maxval=duration_est)
    ev_target = jax.random.uniform(k_t, (k,), minval=rng[ev_param, 0], maxval=rng[ev_param, 1])
    ev_dur = jnp.asarray(TRANSITION_S, dtype=jnp.float32)[ev_param]
    ev_active = jnp.arange(k) < n_events
    return Scenario(base, ev_param, ev_start, ev_dur, ev_target, ev_active)


def schedule(scn: Scenario, n_beats: int):
    """Per-beat values [n_beats + 1, 4] of (rs, hr, emax, vtot) and beat start times [n_beats + 1]."""
    start = jnp.where(scn.ev_active, scn.ev_start, jnp.inf)
    order = jnp.argsort(start)
    param, start, dur, target = scn.ev_param[order], start[order], scn.ev_dur[order], scn.ev_target[order]
    active = jnp.isfinite(start)

    # increment of each event relative to the value left by the previous events on the same parameter
    def inc(cur, ev):
        p, tgt, act = ev
        delta = jnp.where(act, tgt - cur[p], 0.0)
        return cur.at[p].add(delta), delta

    _, delta = jax.lax.scan(inc, scn.base, (param, target, active))
    onehot = jax.nn.one_hot(param, 4, dtype=scn.base.dtype)  # [K, 4]
    start_f = jnp.where(active, start, 0.0)

    def values_at(t):
        frac = jnp.where(active, jnp.clip((t - start_f) / dur, 0.0, 1.0), 0.0)
        return scn.base + (frac * delta) @ onehot

    def beat(t, _):
        v = values_at(t)
        return t + 60.0 / v[HR], (v, t)

    _, (vals, times) = jax.lax.scan(beat, jnp.zeros((), scn.base.dtype), None, length=n_beats + 1)
    return vals, times


def make_scenario(base: dict[str, float], events: list[tuple[str, float, float]], k: int | None = None) -> Scenario:
    """Deterministic scenario. ``events`` = [(param, start_s, target), ...], default durations."""
    k = max(len(events), 1) if k is None else k
    ev = events + [("rs", 0.0, 0.0)] * (k - len(events))
    return Scenario(
        base=jnp.array([base[n] for n in VARYING], dtype=jnp.float32),
        ev_param=jnp.array([VARYING.index(e[0]) for e in ev], dtype=jnp.int32),
        ev_start=jnp.array([e[1] for e in ev], dtype=jnp.float32),
        ev_dur=jnp.array([TRANSITION_S[VARYING.index(e[0])] for e in ev], dtype=jnp.float32),
        ev_target=jnp.array([e[2] for e in ev], dtype=jnp.float32),
        ev_active=jnp.arange(k) < len(events),
    )


# ------------------------------------------------------------------------ stress families (6.5)
def stress_family(key, patient: Patient, kind: str, n: int, k: int = 3) -> Scenario:
    """Randomized stress scenarios (Phase 6.5): a resting patient (Rs, Emax 0.8-1.2 x, Vtot 0.95-1.05 x,
    HR 65-95) and one stress event starting at 60-240 s:
      hypertension  Rs -> 1.5-2.0 x the patient's
      exercise      Rs -> 0.5-0.8 x and HR -> 100-135
      hypovolemia   Vtot -> 0.75-0.88 x (preload drop; suction risk)
    Unused event slots are inactive (``k`` slots for stacking with other scenarios)."""
    kb, kh, ke, kt, k1, k2 = jax.random.split(key, 6)
    u = lambda kk, lo, hi: jax.random.uniform(kk, (n,), minval=lo, maxval=hi)  # noqa: E731
    kb1, kb2, kb3 = jax.random.split(kb, 3)
    base = jnp.stack([u(kb1, 0.8, 1.2) * patient.rs, u(kh, 65.0, 95.0), u(kb2, 0.8, 1.2) * patient.emax,
                      u(kb3, 0.95, 1.05) * patient.vtot], axis=-1)
    t0 = u(kt, 60.0, 240.0)
    if kind == "hypertension":
        params, targets = [RS], [u(k1, 1.5, 2.0) * patient.rs]
    elif kind == "exercise":
        params, targets = [RS, HR], [u(k1, 0.5, 0.8) * patient.rs, u(k2, 100.0, 135.0)]
    elif kind == "hypovolemia":
        params, targets = [VTOT], [u(k1, 0.75, 0.88) * patient.vtot]
    else:
        raise ValueError(kind)
    m = len(params)
    pad = k - m
    ev_param = jnp.asarray(params + [0] * pad, jnp.int32)
    return Scenario(
        base=base.astype(jnp.float32),
        ev_param=jnp.broadcast_to(ev_param, (n, k)),
        ev_start=jnp.concatenate([jnp.stack([t0] * m, -1), jnp.zeros((n, pad))], -1).astype(jnp.float32),
        ev_dur=jnp.broadcast_to(jnp.asarray([TRANSITION_S[p] for p in params] + [1.0] * pad, jnp.float32), (n, k)),
        ev_target=jnp.concatenate([jnp.stack(targets, -1), jnp.zeros((n, pad))], -1).astype(jnp.float32),
        ev_active=jnp.broadcast_to(jnp.arange(k) < m, (n, k)),
    )


# ------------------------------------------------------------------------------ protocols
def protocol_batch(patient: Patient) -> tuple[list[str], Scenario]:
    """The three evaluation protocols stacked into one batch (event slots padded with inactive events)."""
    protos = {"ferreira_exercise": ferreira_exercise, "ferreira_hypertension": ferreira_hypertension, "fz_sequential": fz_sequential}
    scns = [f(patient)[0] for f in protos.values()]
    k_max = max(s.ev_param.shape[0] for s in scns)
    fields = ("ev_param", "ev_start", "ev_dur", "ev_target", "ev_active")

    def pad(s):
        extra = k_max - s.ev_param.shape[0]
        return s._replace(**{f: jnp.concatenate([getattr(s, f), jnp.zeros((extra,), getattr(s, f).dtype)]) for f in fields})

    return list(protos), jax.tree.map(lambda *xs: jnp.stack(xs), *[pad(s) for s in scns])


def ferreira_exercise(patient: Patient) -> tuple[Scenario, float]:
    """Ferreira et al. 2009, Table II scenario 1: baseline (Rs, HR 75) -> light exercise (0.8 Rs,
    HR 90) at 200 s -> strenuous exercise (0.6 Rs, HR 135) at 400 s. Rs scaled to the patient's base.
    Returns the scenario and its duration [s]."""
    b = {"rs": patient.rs, "hr": 75.0, "emax": patient.emax, "vtot": patient.vtot}
    ev = [("rs", 200.0, 0.8 * patient.rs), ("hr", 200.0, 90.0), ("rs", 400.0, 0.6 * patient.rs), ("hr", 400.0, 135.0)]
    return make_scenario(b, ev), 600.0


def ferreira_hypertension(patient: Patient) -> tuple[Scenario, float]:
    """Ferreira et al. 2009, Table II scenario 2: Rs -> 1.8 Rs at 200 s, HR 75."""
    b = {"rs": patient.rs, "hr": 75.0, "emax": patient.emax, "vtot": patient.vtot}
    return make_scenario(b, [("rs", 200.0, 1.8 * patient.rs)]), 400.0


# Fernandez-Zapico et al. 2024, Table 2 ("minimum range" column) as multipliers of their default
# (TBV 5800 ml, HR 80, SVR 0.9, Els 0.8). Our Vtot has no venous compartment, so the TBV drop
# is replaced by a Vtot drop of 15 % (assumption).
FZ_LIMITS = {"vtot": (0.85, 1.0), "hr": (60.0, 150.0), "rs": (0.525 / 0.9, 1.275 / 0.9), "emax": (0.5 / 0.8, 1.1 / 0.8)}


def fz_sequential(patient: Patient, hold_s: float = 20.0) -> tuple[Scenario, float]:
    """Sequential one-parameter-at-a-time protocol in the spirit of Fernandez-Zapico Fig. 2:
    for Vtot, HR, Rs and Emax: go to the upper limit, back to baseline, to the lower limit and back,
    each transition followed by a ``hold_s`` stabilization phase."""
    base = {"rs": patient.rs, "hr": patient.hr, "emax": patient.emax, "vtot": patient.vtot}
    current = dict(base)
    events, t = [], hold_s
    for name in ("vtot", "hr", "rs", "emax"):
        lo, hi = FZ_LIMITS[name]
        up, down = (hi, lo) if name == "hr" else (hi * base[name], lo * base[name])
        for target in (up, base[name], down, base[name]):
            if abs(target - current[name]) < 1e-9:
                continue
            events.append((name, t, target))
            current[name] = target
            t += TRANSITION_S[VARYING.index(name)] + hold_s
    return make_scenario(base, events), t
