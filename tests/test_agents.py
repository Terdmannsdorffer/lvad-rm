"""PPO pieces: GAE against a reference loop, and one tiny training update for each agent type."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lvadrm.agents.ppo import PPOConfig, gae, make_ppo
from lvadrm.agents.specs import SPECS, env_config_for, fz_reward
from lvadrm.env import EnvConfig
from lvadrm.patients import HF_PATIENT


def test_gae_matches_reference():
    rng = np.random.default_rng(0)
    T, B, gamma, lam = 7, 3, 0.9, 0.8
    v, r, last = rng.normal(size=(T, B)), rng.normal(size=(T, B)), rng.normal(size=B)
    adv, tgt = gae(jnp.asarray(v), jnp.asarray(r), jnp.asarray(last), gamma, lam)
    ref = np.zeros((T, B))
    a, v_next = np.zeros(B), last
    for t in reversed(range(T)):
        a = r[t] + gamma * v_next - v[t] + gamma * lam * a
        ref[t], v_next = a, v[t]
    np.testing.assert_allclose(np.asarray(adv), ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(tgt), ref + v, rtol=1e-6, atol=1e-6)


def test_fz_reward():
    obs = jnp.array([[0.5, -0.2], [-0.5, 0.3]])
    np.testing.assert_allclose(np.asarray(fz_reward(obs, jnp.array([0.01, 0.01]))), [1.0 + 0.0, -2.0 + 1.0])
    np.testing.assert_allclose(np.asarray(fz_reward(obs, jnp.array([-0.01, -0.01]))), [-2.0 + 1.0, 1.0 + 0.0])


@pytest.mark.parametrize("agent", ["oracle", "rnn", "fz"])
def test_tiny_training_update(agent):
    spec = SPECS[agent]
    env_cfg = env_config_for(spec, EnvConfig(patient=HF_PATIENT, episode_beats=4, warmup_beats=2, n_inner=20, unroll=1))
    cfg = PPOConfig(n_envs=12, n_steps=2, total_beats=48, n_epochs=2, minibatch_size=5, hidden=16)
    init, update, _ = make_ppo(spec, env_cfg, cfg)
    ts = init(jax.random.PRNGKey(0))
    p0 = jax.tree.leaves(ts.params)[0].copy()
    for _ in range(3):  # crosses an episode boundary (synchronized reset)
        ts, m = update(ts)
    assert all(np.isfinite(float(v)) for v in m.values())
    assert int(ts.env_state.k[0]) == 2  # reset after 4 beats, then 2 more
    assert not np.allclose(np.asarray(p0), np.asarray(jax.tree.leaves(ts.params)[0]))
