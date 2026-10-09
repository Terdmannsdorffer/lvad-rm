"""Beat-by-beat LVAD environment: functional and pure JAX (jit / vmap friendly, gymnax style).

    obs, state = reset(key, cfg)                      # sampled patient + events, warm-up beats
    obs, state, reward, done, info = step(key, state, action, cfg)

One step is one cardiac beat. The action is a multiplicative speed change applied at the start
of the beat (zero-order hold) and clipped to [omega_min, omega_max]. ``step_speed`` takes the
speed directly (used by the classical baselines).

Episodes have a fixed length, so vectorized environments stay synchronized and are reset
together by the training loop (no per-step auto-reset, which would re-run the warm-up).

``info`` carries what the policy must not see: true labels, the RM state before/after the
beat, reward components, hemodynamics and the current patient parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from lvadrm.integrate import SAMPLE_NAMES, simulate_beat
from lvadrm.labels import beat_labels
from lvadrm.model import initial_state, total_volume
from lvadrm.params import make_params
from lvadrm.reward_machine import RMConfig, rm_rewards, sigma_from_labels
from lvadrm.scenarios import EMAX, HR, RS, VTOT, Patient, Scenario, ScenarioConfig, sample_scenario, schedule

QP_IDX = SAMPLE_NAMES.index("qp")


@dataclass(frozen=True)
class EnvConfig:
    patient: Patient
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    rm: RMConfig = field(default_factory=RMConfig)
    episode_beats: int = 300
    warmup_beats: int = 15
    actions: tuple[float, ...] = (-0.05, -0.01, 0.0, 0.01, 0.05)
    omega_min: float = 6000.0
    omega_max: float = 16000.0
    omega0_scale: tuple[float, float] = (0.9, 1.1)  # initial speed = patient.omega * U(scale)
    n_seg: int = 64
    n_inner: int = 105  # N_sub = 6720: dt <= 1.5e-4 s down to HR 60 (Phase 1)
    unroll: int = 8
    snr_db: float | None = 25.0  # uniform noise on the measured pump flow (None: noiseless)
    n_last_actions: int = 4
    obs_mode: str = "pump"  # "pump" (pump signals only) or "pressure" (Fernandez-Zapico MDP)
    eps_avo: float = 0.5  # [ml] aortic valve opening if the ejected volume exceeds it
    back_ml: float = 1.0  # [ml] BACK if the volume regurgitated through the pump in a beat exceeds it
    co_low: float = 4.4  # [l/min] LOW label (cardiac index 2.2 l/min/m2 with BSA 2 m2)
    edp_ref: float = 14.0  # [mmHg] reference EDP of the Fernandez-Zapico observation


class EnvState(NamedTuple):
    y: jax.Array  # [6] model state at the start of the beat
    omega: jax.Array  # current speed [rpm]
    k: jax.Array  # beat index in the episode
    t: jax.Array  # time [s]
    u: jax.Array  # reward machine state
    sched: jax.Array  # [episode_beats + 1, 4] per-beat (rs, hr, emax, vtot)
    last_actions: jax.Array  # [n_last_actions] last fractional speed changes
    prev_pressure: jax.Array  # [2] previous (sEDP, svalve) for the pressure observation


def beat_params(patient: Patient, v):
    """Model parameters for one beat from the schedule row v = (rs, hr, emax, vtot)."""
    return make_params(emax=v[EMAX], emin=patient.emin, v0=patient.v0, hr=v[HR], rs=v[RS])


def _qp_features(key, samples, tc, cfg: EnvConfig):
    """Pump-flow features from the (noisy) sampled waveform of one beat."""
    qp = samples[:, QP_IDX]
    if cfg.snr_db is not None:
        power = jnp.mean(qp**2)
        amp = jnp.sqrt(3.0 * power * 10.0 ** (-cfg.snr_db / 10.0))
        qp = qp + jax.random.uniform(key, qp.shape, minval=-amp, maxval=amp)
    dq = jnp.diff(qp, append=qp[:1]) / (tc / qp.shape[0])
    feats = jnp.stack([qp.mean(), qp.min(), qp.max(), qp.max() - qp.min(), qp.std()]) / 100.0
    return jnp.concatenate([feats, jnp.stack([dq.max(), dq.min()]) / 5000.0]), qp


def _observation(key, state: EnvState, stats, samples, cfg: EnvConfig):
    if cfg.obs_mode == "pump":
        qf, _ = _qp_features(key, samples, stats.tc, cfg)
        speed = jnp.stack([(state.omega - 10000.0) / 2000.0, 60.0 / stats.tc / 100.0])
        return jnp.concatenate([qf, speed, state.last_actions * 20.0])
    if cfg.obs_mode == "pressure":
        cur = jnp.stack([stats.edp - cfg.edp_ref, stats.lvp_max - stats.aop_max]) / 10.0
        return jnp.concatenate([cur, state.prev_pressure])
    raise ValueError(f"unknown obs_mode {cfg.obs_mode!r}")


def obs_size(cfg: EnvConfig) -> int:
    return {"pump": 9 + cfg.n_last_actions, "pressure": 4}[cfg.obs_mode]


def _run_beat(state: EnvState, omega0, omega_rate, omega_hold, cfg: EnvConfig):
    v, v_next = state.sched[state.k], state.sched[state.k + 1]
    p = beat_params(cfg.patient, v)
    tc = 60.0 / v[HR]
    # volume source: reach the scheduled Vtot at the end of the beat (also removes float32 drift)
    q_vol = (v_next[VTOT] - total_volume(state.y, p)) / tc
    return simulate_beat(
        state.y, p, omega0, omega_rate, q_vol, omega_hold, n_seg=cfg.n_seg, n_inner=cfg.n_inner, unroll=cfg.unroll
    )


@partial(jax.jit, static_argnames=("cfg",))
def reset_with_scenario(key, scn: Scenario, cfg: EnvConfig, omega0=None):
    """Reset with a given scenario; ``omega0`` overrides the sampled initial speed."""
    k_w, k_obs = jax.random.split(key)
    sched, _ = schedule(scn, cfg.episode_beats)
    if omega0 is None:
        lo, hi = cfg.omega0_scale
        omega0 = cfg.patient.omega * jax.random.uniform(k_w, (), minval=lo, maxval=hi)
    omega0 = jnp.clip(jnp.asarray(omega0, sched.dtype), cfg.omega_min, cfg.omega_max)
    p0 = beat_params(cfg.patient, sched[0])
    y0 = initial_state(p0, sched[0, VTOT]).astype(sched.dtype)

    # warm-up at constant speed and constant parameters (schedule row 0) towards the periodic regime
    warm = EnvState(y0, omega0, jnp.int32(0), jnp.zeros((), sched.dtype), jnp.int32(0),
                    jnp.broadcast_to(sched[:1], sched.shape), jnp.zeros(cfg.n_last_actions, sched.dtype),
                    jnp.zeros(2, sched.dtype))

    def body(st, _):
        y, stats, samples = _run_beat(st, omega0, 0.0, 0.0, cfg)
        return st._replace(y=y), (stats, samples)

    warm, (stats, samples) = jax.lax.scan(body, warm, None, length=cfg.warmup_beats)
    last = jax.tree.map(lambda x: x[-1], stats)
    pressure = jnp.stack([last.edp - cfg.edp_ref, last.lvp_max - last.aop_max]) / 10.0
    state = warm._replace(sched=sched, prev_pressure=pressure)
    return _observation(k_obs, state, last, samples[-1], cfg), state


@partial(jax.jit, static_argnames=("cfg",))
def reset(key, cfg: EnvConfig, omega0=None):
    k_scn, k_rest = jax.random.split(key)
    scn = sample_scenario(k_scn, cfg.patient, cfg.scenario, cfg.episode_beats)
    return reset_with_scenario(k_rest, scn, cfg, omega0)


@partial(jax.jit, static_argnames=("cfg",))
def step_speed(key, state: EnvState, omega0, cfg: EnvConfig, omega_rate=0.0, omega_hold=0.0, frac=None):
    """Advance one beat with speed omega(t) = omega0 + omega_rate * t inside the beat."""
    omega0 = jnp.clip(jnp.asarray(omega0, state.omega.dtype), cfg.omega_min, cfg.omega_max)
    y, stats, samples = _run_beat(state, omega0, omega_rate, omega_hold, cfg)
    omega_end = jnp.clip(omega0 + omega_rate * stats.tc, cfg.omega_min, cfg.omega_max)

    labels = beat_labels(stats, eps_avo=cfg.eps_avo, back_ml=cfg.back_ml, co_min=cfg.co_low)
    u_next, viol, comps = rm_rewards(state.u, labels.avo, labels.suc, labels.back, stats.co, state.omega, stats.omega_mean, cfg.rm)
    if frac is None:
        frac = omega0 / state.omega - 1.0
    k_next = state.k + 1
    done = k_next >= cfg.episode_beats
    if cfg.rm.terminal_on_violation:
        done = done | viol

    pressure = jnp.stack([stats.edp - cfg.edp_ref, stats.lvp_max - stats.aop_max]) / 10.0
    new_state = EnvState(
        y=y,
        omega=omega_end,
        k=k_next,
        t=state.t + stats.tc,
        u=u_next,
        sched=state.sched,
        last_actions=jnp.concatenate([jnp.atleast_1d(frac).astype(state.last_actions.dtype), state.last_actions[:-1]]),
        prev_pressure=state.prev_pressure,
    )
    obs = _observation(key, new_state, stats, samples, cfg)
    new_state = new_state._replace(prev_pressure=pressure)
    v = state.sched[state.k]
    info = {
        "labels": jnp.stack(list(labels)),
        "sigma": sigma_from_labels(*labels),
        "u": state.u,
        "u_next": u_next,
        "violation": viol,
        "reward_components": comps,
        "omega": stats.omega_mean,
        "co": stats.co,
        "qlv": stats.qlv,
        "v_aortic": stats.v_aortic,
        "v_back": stats.v_back,
        "edv": stats.edv,
        "edp": stats.edp,
        "lvp_min": stats.lvp_min,
        "lvp_max": stats.lvp_max,
        "aop_max": stats.aop_max,
        "aop_mean": stats.aop_mean,
        "lap_mean": stats.lap_mean,
        "qp_min": stats.qp_min,
        "qp_mean": stats.qp_mean,
        "t_qp_min": stats.t_qp_min,
        "tc": stats.tc,
        "vtot": stats.vtot,
        "params": v,
    }
    return obs, new_state, comps.total(), done, info


@partial(jax.jit, static_argnames=("cfg",))
def step(key, state: EnvState, action, cfg: EnvConfig):
    frac = jnp.asarray(cfg.actions, state.omega.dtype)[action]
    return step_speed(key, state, state.omega * (1.0 + frac), cfg, frac=frac)
