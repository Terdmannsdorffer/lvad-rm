"""Belief updates, synthetic detectors and a toy TDM."""

import jax
import jax.numpy as jnp
import numpy as np

from lvadrm.detectors import SynState, SyntheticDetector, synthetic_errors
from lvadrm.inference import ibu_update, initial_belief, log_likelihood, naive_update, tdm_input, train_tdm


def test_ibu_example_5_2():
    """Li et al. (2024) Example 5.2: an event with probability 0.3 queried k times has been seen with
    probability 1 - 0.7^k (counting RM long enough that no violation happens)."""
    n = 12
    b = initial_belief(n)
    for k in range(1, 8):
        b = ibu_update(b, 0.3, n)
        np.testing.assert_allclose(float(b.sum()), 1.0, rtol=1e-6)
        np.testing.assert_allclose(float(b[k]), 0.7**k, rtol=1e-5)  # never opened in k beats
        np.testing.assert_allclose(1.0 - float(b[k]), 1.0 - 0.7**k, rtol=1e-5)


def test_ibu_with_certain_detections_equals_naive():
    n = 4
    b, u = initial_belief(n), jnp.int32(0)
    for d in [0, 0, 0, 0, 1, 0, 0]:
        b = ibu_update(b, float(d), n)
        u = naive_update(u, jnp.bool_(d), n)
        np.testing.assert_allclose(np.asarray(b), np.asarray(jax.nn.one_hot(u, n)), atol=1e-6)


def test_log_likelihood_floor():
    b = jnp.array([[1.0, 0.0, 0.0], [0.2, 0.5, 0.3]])
    ll = log_likelihood(b, jnp.array([1, 1]))
    np.testing.assert_allclose(np.asarray(ll), [np.log(0.01), np.log(0.5)], rtol=1e-6)


def test_markov_detector_rate_and_persistence():
    det = SyntheticDetector(kind="markov", eps=0.2, rho=0.9)
    keys = jax.random.split(jax.random.PRNGKey(0), 200_000)

    def body(ds, k):
        err, ds = synthetic_errors(k, ds, jnp.bool_(True), 5.0, det)
        return ds, err

    _, err = jax.lax.scan(body, SynState(jnp.bool_(False), jnp.float32(0.2)), keys)
    err = np.asarray(err)
    assert abs(err.mean() - 0.2) < 0.01
    assert abs(err[1:][err[:-1]].mean() - 0.9) < 0.01


def test_tdm_converges_on_toy_problem():
    """Random openings (p = 0.5), detections with 10 % independent errors: the TDM should reach the
    belief quality of IBU, which is optimal here."""
    rng = np.random.default_rng(0)
    n, E, T, eps = 3, 384, 30, 0.1
    avo = rng.random((E, T)) < 0.5
    d = avo ^ (rng.random((E, T)) < eps)
    p = np.where(d, 0.9, 0.1).astype(np.float32)  # calibrated: P(avo | d) = 1 - eps with prior 0.5
    u = np.zeros((E, T), dtype=np.int32)
    cur = np.zeros(E, dtype=int)
    for t in range(T):
        viol = ~avo[:, t] & (cur == n - 1)
        cur = np.where(avo[:, t] | viol, 0, cur + 1)
        u[:, t] = cur
    x = np.asarray(tdm_input(jnp.asarray(p), jnp.zeros((E, T, 1))))
    model, params, hist = train_tdm(x[:320], u[:320], x[320:], u[320:], n, hidden=32, epochs=25, batch_episodes=32, log=None)
    b = initial_belief(n)
    lls = []
    for t in range(T):
        b = ibu_update(jnp.broadcast_to(b, (E - 320, n)) if t == 0 else b, jnp.asarray(p[320:, t]), n)
        lls.append(np.asarray(log_likelihood(b, jnp.asarray(u[320:, t]))))
    ibu_ll = float(np.mean(lls))
    assert hist[-1]["val_ll"] > ibu_ll - 0.05, (hist[-1]["val_ll"], ibu_ll)


def test_class_conditional_markov_rates():
    det = SyntheticDetector(kind="markov", eps=0.14, eps_fp=0.07, rho=0.9)
    assert det.name == "markov_fn0.14_fp0.07_r0.9"
    for avo, target in ((True, 0.14), (False, 0.07)):
        keys = jax.random.split(jax.random.PRNGKey(1), 200_000)

        def body(ds, k, avo=avo):
            err, ds = synthetic_errors(k, ds, jnp.bool_(avo), 5.0, det)
            return ds, err

        _, err = jax.lax.scan(body, SynState(jnp.bool_(False), jnp.float32(0.1)), keys)
        assert abs(float(np.asarray(err).mean()) - target) < 0.01
