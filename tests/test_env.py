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
