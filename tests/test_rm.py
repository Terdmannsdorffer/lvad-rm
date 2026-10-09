"""Reward machine transitions and rewards."""

import jax.numpy as jnp
import numpy as np

from lvadrm.reward_machine import RMConfig, rm_rewards, rm_transition, sigma_from_labels, transition_table, violation_table


def run(avo_seq, n):
    u, out = jnp.int32(0), []
    for a in avo_seq:
        u, viol = rm_transition(u, jnp.bool_(a), n)
        out.append((int(u), bool(viol)))
    return out


def test_counting_transitions():
    # N = 3: two closed beats are allowed, the third consecutive one is a violation
    assert run([0, 0, 0, 0, 1, 0], 3) == [(1, False), (2, False), (0, True), (1, False), (0, False), (1, False)]
    # an opening at u_{N-1} avoids the violation
    assert run([0, 0, 1], 3) == [(1, False), (2, False), (0, False)]


def test_tables_match_transition():
    n = 5
    nxt, vio = transition_table(n), violation_table(n)
    for u in range(n):
        for sigma in range(16):
            un, v = rm_transition(jnp.int32(u), jnp.bool_(sigma & 1), n)
            assert int(un) == nxt[u, sigma] and bool(v) == vio[u, sigma]


def test_rewards():
    cfg = RMConfig(n=3)
    u, viol, r = rm_rewards(jnp.int32(2), jnp.bool_(False), jnp.bool_(True), jnp.bool_(True), jnp.float32(4.0), 10000.0, 10100.0, cfg)
    assert int(u) == 0 and bool(viol)
    np.testing.assert_allclose(float(r.viol), cfg.r_viol)
    np.testing.assert_allclose(float(r.suc), cfg.r_suc)
    np.testing.assert_allclose(float(r.back), cfg.r_back)
    np.testing.assert_allclose(float(r.perf), -cfg.w_perf * 0.2, rtol=1e-6)
    np.testing.assert_allclose(float(r.act), -cfg.w_act * 0.01, rtol=1e-6)
    np.testing.assert_allclose(float(r.total()), cfg.r_viol + cfg.r_suc + cfg.r_back - 0.02 - 1e-4, rtol=1e-6)


def test_sigma_encoding():
    s = sigma_from_labels(jnp.bool_(True), jnp.bool_(False), jnp.bool_(True), jnp.bool_(True))
    assert int(s) == 1 + 4 + 8
