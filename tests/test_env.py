"""Environment: shapes, determinism, volume schedule, RM bookkeeping, scenarios."""

import jax
import jax.numpy as jnp
import numpy as np

from lvadrm.env import EnvConfig, obs_size, reset, reset_with_scenario, step
from lvadrm.patients import HF_PATIENT
from lvadrm.reward_machine import RMConfig, rm_transition
from lvadrm.scenarios import HR, VTOT, ScenarioConfig, fz_sequential, make_scenario, sample_scenario, schedule

CFG = EnvConfig(patient=HF_PATIENT, episode_beats=12, warmup_beats=5, n_inner=40, unroll=1, rm=RMConfig(n=3))


def test_reset_and_step_shapes_and_determinism():
    obs, st = reset(jax.random.PRNGKey(0), CFG)
    obs2, _ = reset(jax.random.PRNGKey(0), CFG)
    assert obs.shape == (obs_size(CFG),) and bool(jnp.all(jnp.isfinite(obs)))
    np.testing.assert_array_equal(np.asarray(obs), np.asarray(obs2))
    assert st.sched.shape == (CFG.episode_beats + 1, 4) and int(st.k) == 0 and int(st.u) == 0
    obs, st2, r, done, info = step(jax.random.PRNGKey(1), st, 2, CFG)
    assert obs.shape == (obs_size(CFG),) and np.isfinite(float(r)) and not bool(done)
    assert int(st2.k) == 1 and info["labels"].shape == (4,)
    assert abs(float(st2.omega) - float(st.omega)) < 1e-3 * float(st.omega)  # action 2 = keep speed


def test_episode_rm_bookkeeping_and_volume_schedule():
    _, st = reset(jax.random.PRNGKey(3), CFG)
    keys = jax.random.split(jax.random.PRNGKey(4), CFG.episode_beats)
    done = False
    for k in range(CFG.episode_beats):
        _, st_next, r, done, info = step(keys[k], st, k % 5, CFG)
        u_expected, viol = rm_transition(st.u, info["labels"][0], CFG.rm.n)
        assert int(info["u_next"]) == int(u_expected) == int(st_next.u)
        assert bool(info["violation"]) == bool(viol)
        np.testing.assert_allclose(float(r), float(info["reward_components"].total()), rtol=1e-6)
        # the volume source makes Vtot follow the schedule beat by beat
        np.testing.assert_allclose(float(info["vtot"]), float(st.sched[k + 1, VTOT]), rtol=1e-4)
        st = st_next
    assert bool(done)


def test_schedule_ramps():
    base = {"rs": 1.0, "hr": 60.0, "emax": 1.0, "vtot": 300.0}
    scn = make_scenario(base, [("rs", 10.0, 2.0), ("hr", 20.0, 120.0)])
    vals, times = schedule(scn, 160)  # ~95 s (HR 60 -> 120)
    assert times[-1] > 80.0
    vals, times = np.asarray(vals), np.asarray(times)
    rs_at = np.interp([5.0, 35.0, 60.0, 75.0], times, vals[:, 0])
    np.testing.assert_allclose(rs_at, [1.0, 1.5, 2.0, 2.0], atol=0.03)  # 50 s ramp from t = 10 s
    assert vals[:, HR].max() <= 120.0 + 1e-4 and abs(vals[-1, HR] - 120.0) < 1e-4  # 10 s ramp
    assert np.all(np.diff(times) > 0)


def test_sampled_scenarios_stay_in_range():
    cfg = ScenarioConfig()
    scn = jax.vmap(lambda k: sample_scenario(k, HF_PATIENT, cfg, 300))(jax.random.split(jax.random.PRNGKey(0), 256))
    vals, _ = jax.vmap(lambda s: schedule(s, 300))(scn)
    vals = np.asarray(vals)
    rs_lo, rs_hi = cfg.rs_scale[0] * HF_PATIENT.rs, cfg.rs_scale[1] * HF_PATIENT.rs
    assert vals[..., 0].min() >= rs_lo - 1e-5 and vals[..., 0].max() <= rs_hi + 1e-5
    assert vals[..., HR].min() >= cfg.hr_range[0] - 1e-4 and vals[..., HR].max() <= cfg.hr_range[1] + 1e-4
    assert vals[..., VTOT].min() >= cfg.vtot_scale[0] * HF_PATIENT.vtot - 1e-3


def test_protocol_scenario_runs():
    scn, duration = fz_sequential(HF_PATIENT)
    assert 600.0 < duration < 1000.0
    cfg = EnvConfig(patient=HF_PATIENT, episode_beats=3, warmup_beats=2, n_inner=40, unroll=1)
    obs, st = reset_with_scenario(jax.random.PRNGKey(0), scn, cfg, omega0=HF_PATIENT.omega)
    assert abs(float(st.omega) - HF_PATIENT.omega) < 1.0 and bool(jnp.all(jnp.isfinite(obs)))


def test_observation_modes():
    for mode in ("pump", "pressure", "pressure_plus"):
        cfg = EnvConfig(patient=HF_PATIENT, episode_beats=2, warmup_beats=2, n_inner=20, unroll=1, obs_mode=mode)
        obs, st = reset(jax.random.PRNGKey(0), cfg)
        obs2, *_ = step(jax.random.PRNGKey(1), st, 0, cfg)
        assert obs.shape == obs2.shape == (obs_size(cfg),)
        assert bool(jnp.all(jnp.isfinite(obs2)))


def test_estimated_flow_observation_and_starling_target():
    """Phase 6.5 options: per-episode flow errors stay fixed, the observation differs from the measured
    mode, and the reward uses the preload-dependent target."""
    from dataclasses import replace

    from lvadrm.env import starling_target

    cfg = replace(CFG, obs_model="estimated", perf_target="starling", flow_noise_scale=2.0)
    obs, st = reset(jax.random.PRNGKey(0), cfg)
    obs_m, st_m = reset(jax.random.PRNGKey(0), CFG)
    assert float(jnp.abs(st.flow_err).sum()) > 0 and not np.allclose(np.asarray(obs), np.asarray(obs_m))
    np.testing.assert_array_equal(np.asarray(st.y), np.asarray(st_m.y))  # same physiology
    obs2, st2, r, _, info = step(jax.random.PRNGKey(1), st, 2, cfg)
    np.testing.assert_array_equal(np.asarray(st2.flow_err), np.asarray(st.flow_err))
    np.testing.assert_allclose(float(info["co_target"]), float(starling_target(info["edp"], cfg)), rtol=1e-6)
    expected = -cfg.rm.w_perf * abs(float(info["co"]) - float(info["co_target"])) / float(info["co_target"])
    np.testing.assert_allclose(float(info["reward_components"].perf), expected, rtol=1e-5)


def test_stress_family_and_closure_runs():
    from lvadrm.evaluation import closure_runs
    from lvadrm.scenarios import RS, stress_family

    fam = stress_family(jax.random.PRNGKey(0), HF_PATIENT, "hypertension", 8)
    assert fam.base.shape == (8, 4) and fam.ev_param.shape == (8, 3)
    assert bool(jnp.all(fam.ev_param[:, 0] == RS)) and bool(jnp.all(fam.ev_target[:, 0] >= 1.5 * HF_PATIENT.rs))
    assert int(fam.ev_active.sum()) == 8
    avo = np.array([[1, 0, 0, 1, 0]], dtype=bool).T
    tc = np.full((5, 1), 0.5)
    np.testing.assert_allclose(closure_runs(avo, tc)[0], [1.0, 0.5])
