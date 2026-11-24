import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lvadrm.estimators import FilterModel, LambdaDetector, filter_belief, fit_detector, make_estimator
from lvadrm.inference import ibu_update, rm_state_sequence
from lvadrm.presets import env_v3
from lvadrm.reward_machine import rm_transition

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def test_saturating_rm_pays_every_closed_beat():
    n, u, viol = 4, jnp.int32(0), []
    for avo in [False] * 6 + [True, False]:
        u, v = rm_transition(u, jnp.bool_(avo), n, True)
        viol.append(bool(v))
    assert viol == [False, False, False, True, True, True, False, False]
    assert int(u) == 1


def test_rm_state_sequence_matches_transition():
    rng = np.random.default_rng(0)
    avo = rng.random((5, 60)) < 0.15
    seq = rm_state_sequence(avo, 5, True)
    for e in range(5):
        u = jnp.int32(0)
        for t in range(60):
            u, _ = rm_transition(u, jnp.bool_(avo[e, t]), 5, True)
            assert int(u) == seq[e, t]


def test_saturating_ibu_is_a_product_of_marginals():
    n = 6
    b = jnp.zeros(n).at[0].set(1.0)
    p = jnp.array([0.3, 0.1, 0.4, 0.2, 0.25, 0.15, 0.3])
    for pi in p:
        b = ibu_update(b, pi, n, True)
        assert np.isclose(float(b.sum()), 1.0)
    for k in range(1, n):  # P(u >= k) = product of the last k probabilities of no event
        assert np.isclose(float(b[k:].sum()), float(jnp.prod(1 - p[-k:])))


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.9])
def test_lambda_detector_keeps_rates(lam):
    rng = np.random.default_rng(1)
    avo = rng.random(40000) < 0.4
    t_open = np.where(avo, rng.normal(0.11, 0.02, avo.size), 0.0)
    tau, sigma = fit_detector(t_open, avo)
    det = LambdaDetector(tau=tau, sigma=sigma, lam=lam)
    keys = jax.random.split(jax.random.PRNGKey(0), avo.size)
    b = jax.vmap(det.init)(keys)  # one episode per beat: marginal rates
    d = np.asarray(jax.vmap(det.detect)(jax.vmap(lambda k: jax.random.fold_in(k, 1))(keys), b, jnp.asarray(avo), jnp.asarray(t_open)))
    assert abs(d[avo].mean() - 0.86) < 0.015
    assert abs(1 - d[~avo].mean() - 0.93) < 0.015


def _toy_filter():
    x = np.linspace(-0.2, 0.2, 41)
    log_m = np.linspace(np.log(7000), np.log(9000), 15)
    return FilterModel(x=x, p_open=np.clip((x > 0).astype(float), 0.01, 0.99), t_ms=np.full(41, 110.0), alpha=1.0, sigma_t=5.0, drift=0.0,
                       log_m=log_m, prior_m=np.full(15, 1 / 15))


def test_filter_belief_is_normalized_and_counts():
    det = LambdaDetector(tau=56.0, sigma=20.0, lam=0.5)
    n = 5
    est = filter_belief(det, n, _toy_filter())
    obs = jnp.zeros(7).at[1].set((9500.0 - 10000.0) / 2000.0)  # above every threshold: the valve never opens
    bs = est.init(jax.random.PRNGKey(0))
    for t in range(8):
        bs = est.update(jax.random.PRNGKey(t), bs, obs, {"labels": jnp.array([False]), "t_open": jnp.zeros(())})
        v = est.vector(bs)
        assert np.isclose(float(v[:n].sum()), 1.0, atol=1e-5)
    assert float(v[n - 1]) > 0.9


def test_true_and_naive_estimators_agree_with_a_perfect_detector():
    det = LambdaDetector(perfect=True, p_pos=1.0, p_neg=0.0)
    n = 4
    a, b = make_estimator("true", det, n), make_estimator("naive", det, n)
    sa, sb, u = a.init(jax.random.PRNGKey(0)), b.init(jax.random.PRNGKey(0)), jnp.int32(0)
    for t, avo in enumerate([True, False, False, False, False, True, False]):
        u, _ = rm_transition(u, jnp.bool_(avo), n, True)
        info = {"labels": jnp.array([avo]), "t_open": jnp.asarray(0.1 * avo), "u_next": u}
        sa = a.update(jax.random.PRNGKey(t), sa, jnp.zeros(7), info)
        sb = b.update(jax.random.PRNGKey(t), sb, jnp.zeros(7), info)
        np.testing.assert_allclose(a.vector(sa), b.vector(sb))


def test_env_v3_observation_has_no_waveform_statistics():
    from lvadrm.env import obs_size

    assert obs_size(env_v3()) == 7


def test_identity_of_theorem_1():
    from phase7_theory import history, ibu, truth

    for k, s, lam in ((4, 0.8, 0.5), (6, 0.9, 0.9), (3, 0.5, 0.0)):
        e, rhs = history(k, s, lam)
        assert abs(truth(k, s) - e - rhs) < 1e-12
    assert ibu(6, 0.9, 0.0) < truth(6, 0.9)  # persistent labels: underestimates
    assert ibu(6, 0.5, 0.8) > truth(6, 0.5)  # persistent errors: overestimates
