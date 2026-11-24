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
PUMP_FLOW_FEATURES = 7  # the "pump" observation starts with 7 pump-flow features, then speed, HR, last actions


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
    obs_mode: str = "pump"  # "pump", "pressure" (Fernandez-Zapico MDP), "pressure_plus", "reduced", "reduced_pressure"
    eps_avo: float = 0.5  # [ml] aortic valve opening if the ejected volume exceeds it
    back_ml: float = 1.0  # [ml] BACK if the volume regurgitated through the pump in a beat exceeds it
    co_low: float = 4.4  # [l/min] LOW label (cardiac index 2.2 l/min/m2 with BSA 2 m2)
    edp_ref: float = 14.0  # [mmHg] reference EDP of the Fernandez-Zapico observation
    # Phase 6.5. Pump-flow observation: "measured" (true flow + uniform noise at snr_db) or "estimated"
    # (flow estimated from motor current and speed: per-patient gain and offset errors, waveform noise
    # and the estimator's limited bandwidth; magnitudes from published HVAD / Karantonis errors, all
    # multiplied by flow_noise_scale, which is calibrated so that AVO detection is clinical-level).
    obs_model: str = "measured"
    flow_noise_scale: float = 1.0
    flow_gain_sd: float = 0.06  # relative error of the estimated flow (Karantonis et al. 2007: 5.77 %)
    flow_bias_sd: float = 5.0  # [ml/s] offset (HVAD estimator: 0.1 +- 0.3 l/min)
    flow_wave_sd: float = 11.7  # [ml/s] waveform error after smoothing (HVAD: ~0.7 l/min spread)
    flow_smooth: int = 4  # [samples] moving-average window of the estimator (64 samples per beat)
    # Performance target: "fixed" (rm.co_target) or "starling": co_target = clip(co_rest + slope *
    # (EDP - edp_rest), co_lo, co_hi), a preload-sensitive target in the spirit of Starling-like
    # controllers (Mansouri et al. 2017). Values from scripts/phase6_5.py calibrate-starling.
    perf_target: str = "fixed"
    starling: tuple[float, float, float, float, float] = (5.0, 10.0, 0.25, 3.5, 8.0)  # co_rest, edp_rest, slope, lo, hi


class EnvState(NamedTuple):
    y: jax.Array  # [6] model state at the start of the beat
    omega: jax.Array  # current speed [rpm]
    k: jax.Array  # beat index in the episode
    t: jax.Array  # time [s]
    u: jax.Array  # reward machine state
    sched: jax.Array  # [episode_beats + 1, 4] per-beat (rs, hr, emax, vtot)
    last_actions: jax.Array  # [n_last_actions] last fractional speed changes
    prev_pressure: jax.Array  # [2] previous (sEDP, svalve) for the pressure observation
    flow_err: jax.Array  # [2] (relative gain error, offset [ml/s]) of the estimated flow, fixed per episode


def beat_params(patient: Patient, v):
    """Model parameters for one beat from the schedule row v = (rs, hr, emax, vtot)."""
    return make_params(emax=v[EMAX], emin=patient.emin, v0=patient.v0, hr=v[HR], rs=v[RS])


def starling_target(edp, cfg: EnvConfig):
    """Preload-sensitive cardiac-output target [l/min] (perf_target = "starling")."""
    co_rest, edp_rest, slope, lo, hi = cfg.starling
    return jnp.clip(co_rest + slope * (edp - edp_rest), lo, hi)


def sample_flow_err(key, cfg: EnvConfig, dtype=jnp.float32):
    """Per-episode (gain error, offset) of the estimated pump flow."""
    z = jax.random.normal(key, (2,), dtype)
    return z * jnp.asarray([cfg.flow_gain_sd, cfg.flow_bias_sd], dtype) * cfg.flow_noise_scale


def _measured_flow(key, qp, flow_err, cfg: EnvConfig):
    """The pump-flow waveform available to the controller."""
    if cfg.obs_model == "estimated":
        L = cfg.flow_smooth
        noise = jax.random.normal(key, qp.shape, qp.dtype) * cfg.flow_wave_sd * cfg.flow_noise_scale * jnp.sqrt(L)
        raw = qp * (1.0 + flow_err[0]) + flow_err[1] + noise
        idx = (jnp.arange(qp.shape[0])[:, None] - jnp.arange(L)[None, :]) % qp.shape[0]  # causal circular window
        return raw[idx].mean(axis=1)
    if cfg.obs_model != "measured":
        raise ValueError(f"unknown obs_model {cfg.obs_model!r}")
    if cfg.snr_db is not None:
        power = jnp.mean(qp**2)
        amp = jnp.sqrt(3.0 * power * 10.0 ** (-cfg.snr_db / 10.0))
        qp = qp + jax.random.uniform(key, qp.shape, minval=-amp, maxval=amp)
    return qp


def _qp_features(key, samples, tc, cfg: EnvConfig, flow_err=None):
    """Pump-flow features from the measured (or estimated) sampled waveform of one beat."""
    qp = _measured_flow(key, samples[:, QP_IDX], flow_err, cfg)
    dq = jnp.diff(qp, append=qp[:1]) / (tc / qp.shape[0])
    feats = jnp.stack([qp.mean(), qp.min(), qp.max(), qp.max() - qp.min(), qp.std()]) / 100.0
    return jnp.concatenate([feats, jnp.stack([dq.max(), dq.min()]) / 5000.0]), qp


def _observation(key, state: EnvState, stats, samples, cfg: EnvConfig):
    """Returns (observation, measured pump-flow waveform / 100). The waveform is observable in every
    mode (it is what the AVO detectors use) but only the "pump" observation is built from it."""
    qf, qp = _qp_features(key, samples, stats.tc, cfg, state.flow_err)
    speed = jnp.stack([(state.omega - 10000.0) / 2000.0, 60.0 / stats.tc / 100.0])
    if cfg.obs_mode == "reduced":  # Phase 7: mean flow only, so that the AVO information comes from the detector
        return jnp.concatenate([qf[:1], speed, state.last_actions * 20.0]), qp / 100.0
    if cfg.obs_mode == "pump":
        return jnp.concatenate([qf, speed, state.last_actions * 20.0]), qp / 100.0
    cur = jnp.stack([stats.edp - cfg.edp_ref, stats.lvp_max - stats.aop_max]) / 10.0
    if cfg.obs_mode == "reduced_pressure":  # "reduced" plus the pressures of the last two beats (invasive reference)
        return jnp.concatenate([qf[:1], speed, state.last_actions * 20.0, cur, state.prev_pressure]), qp / 100.0
    if cfg.obs_mode == "pressure":
        return jnp.concatenate([cur, state.prev_pressure]), qp / 100.0
    if cfg.obs_mode == "pressure_plus":  # the "pump" observation with pressures instead of flow features
        return jnp.concatenate([cur, state.prev_pressure, speed, state.last_actions * 20.0]), qp / 100.0
    raise ValueError(f"unknown obs_mode {cfg.obs_mode!r}")


def obs_size(cfg: EnvConfig) -> int:
    n = cfg.n_last_actions
    return {"reduced": 3 + n, "reduced_pressure": 7 + n, "pump": 9 + n, "pressure": 4, "pressure_plus": 6 + n}[cfg.obs_mode]


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
    k_flow = jax.random.fold_in(key, 1)  # separate stream: keeps the keys of the "measured" mode unchanged
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
                    jnp.zeros(2, sched.dtype), sample_flow_err(k_flow, cfg, sched.dtype))

    def body(st, _):
        y, stats, samples = _run_beat(st, omega0, 0.0, 0.0, cfg)
        return st._replace(y=y), (stats, samples)

    warm, (stats, samples) = jax.lax.scan(body, warm, None, length=cfg.warmup_beats)
    last = jax.tree.map(lambda x: x[-1], stats)
    pressure = jnp.stack([last.edp - cfg.edp_ref, last.lvp_max - last.aop_max]) / 10.0
    state = warm._replace(sched=sched, prev_pressure=pressure)
    return _observation(k_obs, state, last, samples[-1], cfg)[0], state


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
    if cfg.perf_target == "starling":
        co_target = starling_target(stats.edp, cfg)
    elif cfg.perf_target == "fixed":
        co_target = jnp.full_like(stats.co, cfg.rm.co_target)
    else:
        raise ValueError(f"unknown perf_target {cfg.perf_target!r}")
    u_next, viol, comps = rm_rewards(state.u, labels.avo, labels.suc, labels.back, stats.co, state.omega, stats.omega_mean,
                                     cfg.rm, co_target=co_target)
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
        flow_err=state.flow_err,
    )
    obs, qp_wave = _observation(key, new_state, stats, samples, cfg)
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
        "co_target": co_target,
        "qlv": stats.qlv,
        "v_aortic": stats.v_aortic,
        "t_open": stats.t_open,
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
        "qp_wave": qp_wave,  # measured (noisy) pump flow of the beat / 100, shape [n_seg]
    }
    return obs, new_state, comps.total(), done, info


@partial(jax.jit, static_argnames=("cfg",))
def step(key, state: EnvState, action, cfg: EnvConfig):
    frac = jnp.asarray(cfg.actions, state.omega.dtype)[action]
    return step_speed(key, state, state.omega * (1.0 + frac), cfg, frac=frac)
