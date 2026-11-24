"""Inference of the reward-machine state from noisy AVO detections (Li et al. 2024; HANDOFF section 10).

Naive  feeds binary detections through delta_u: u_hat <- delta(u_hat, d). One error propagates.
IBU    exact belief update assuming independent detector errors across beats:
           b'[u] = sum_{sigma, u'} 1[delta(u', sigma) = u] b[u'] M(h)[sigma]
       Only AVO drives the counting RM, so M(h) reduces to p = P(AVO). Overconfident when errors are
       correlated in time.
TDM    a recurrent network that maps the history of observations and detector outputs directly to a
       distribution over RM states, trained with the true states (consistent under correlation).

Belief quality: mean log-likelihood of the true RM state with probabilities floored at 0.01.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from lvadrm.reward_machine import rm_transition

LL_FLOOR = 0.01


def naive_update(u_hat, d, n: int, saturate: bool = False):
    return rm_transition(u_hat, d, n, saturate)[0]


def ibu_update(b, p, n: int, saturate: bool = False):
    """One belief update for the counting RM with P(AVO) = p. b has shape [..., n]."""
    p = jnp.asarray(p)[..., None]
    if saturate:
        shifted = b[..., : n - 1] * (1.0 - p)
        shifted = shifted.at[..., -1].add(b[..., n - 1] * (1.0 - p[..., 0]))
        return jnp.concatenate([(b * p).sum(-1, keepdims=True), shifted], axis=-1)
    to_zero = (b * p).sum(-1, keepdims=True) + b[..., n - 1 :] * (1.0 - p)  # opening, or violation reset
    shifted = b[..., : n - 1] * (1.0 - p)  # no opening: u -> u + 1
    return jnp.concatenate([to_zero, shifted], axis=-1)


def initial_belief(n: int, dtype=jnp.float32):
    return jax.nn.one_hot(0, n, dtype=dtype)


def rm_state_sequence(avo: np.ndarray, n: int, saturate: bool = False) -> np.ndarray:
    """True counting-RM state after each beat for AVO sequences [E, T] (start in u_0)."""
    u = np.zeros(avo.shape[0], dtype=np.int32)
    out = np.zeros(avo.shape, dtype=np.int32)
    for t in range(avo.shape[1]):
        viol = ~avo[:, t] & (u == n - 1)
        u = np.where(avo[:, t], 0, np.minimum(u + 1, n - 1)) if saturate else np.where(avo[:, t] | viol, 0, u + 1)
        out[:, t] = u
    return out


def log_likelihood(b, u_true, floor: float = LL_FLOOR):
    """log max(b[u_true], floor), elementwise over leading axes."""
    pb = jnp.take_along_axis(b, jnp.asarray(u_true)[..., None], axis=-1)[..., 0]
    return jnp.log(jnp.maximum(pb, floor))


# ----------------------------------------------------------------------------------------------- TDM
class TDM(nn.Module):
    """Encoder -> GRU -> logits over RM states. One step: (h, x) -> (h', logits)."""

    n_states: int
    hidden: int = 128

    @nn.compact
    def __call__(self, h, x):
        e = nn.relu(nn.Dense(self.hidden)(x))
        e = nn.relu(nn.Dense(self.hidden)(e))
        h, y = nn.GRUCell(features=self.hidden)(h, e)
        return h, nn.Dense(self.n_states)(y)


def tdm_input(p, obs_next, features: str = "full"):
    """TDM input for one beat: detector probability (and its logit) plus the agent's observation
    ("full") or only its non-flow part (speed, HR, last actions; "detector"). The "detector" variant sees
    the same information about AVO as IBU, so it isolates what is gained from the temporal structure."""
    from lvadrm.env import PUMP_FLOW_FEATURES

    p = jnp.clip(p, 1e-4, 1.0 - 1e-4)
    pp = jnp.stack([p, jnp.log(p) - jnp.log1p(-p)], axis=-1) * jnp.array([1.0, 0.1])
    extra = obs_next if features == "full" else obs_next[..., PUMP_FLOW_FEATURES:]
    return jnp.concatenate([pp, extra], axis=-1)


def tdm_sequence_logits(model: TDM, params, x):
    """Run the TDM over sequences x [T, B, d] from a zero state. Returns logits [T, B, n]."""
    h0 = jnp.zeros((x.shape[1], model.hidden), x.dtype)

    def body(h, xt):
        h, logits = model.apply(params, h, xt)
        return h, logits

    return jax.lax.scan(body, h0, x)[1]


def train_tdm(x_train, u_train, x_val, u_val, n_states: int, *, hidden: int = 128, epochs: int = 30,
              batch_episodes: int = 64, lr: float = 1e-3, seed: int = 0, log=print):
    """Supervised training of the TDM on full episodes. x: [E, T, d], u: [E, T] (true RM state after
    each beat). Returns (model, params, history)."""
    model = TDM(n_states=n_states, hidden=hidden)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros((1, hidden)), jnp.asarray(x_train[:1, 0]))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = tx.init(params)

    def loss_fn(params, x, u):  # x [B, T, d]
        logits = tdm_sequence_logits(model, params, jnp.swapaxes(x, 0, 1))
        logp = jax.nn.log_softmax(logits)
        return -jnp.take_along_axis(logp, jnp.swapaxes(u, 0, 1)[..., None], axis=-1).mean()

    @jax.jit
    def train_step(params, opt_state, x, u):
        loss, g = jax.value_and_grad(loss_fn)(params, x, u)
        upd, opt_state = tx.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    @jax.jit
    def val_ll(params, x, u):
        b = jax.nn.softmax(tdm_sequence_logits(model, params, jnp.swapaxes(x, 0, 1)))
        return log_likelihood(b, jnp.swapaxes(u, 0, 1)).mean()

    rng = np.random.default_rng(seed)
    history, best = [], (-np.inf, params)
    n_ep = x_train.shape[0]
    for ep in range(epochs):
        perm = rng.permutation(n_ep)
        losses = []
        for i in range(0, n_ep - batch_episodes + 1, batch_episodes):
            idx = perm[i : i + batch_episodes]
            params, opt_state, loss = train_step(params, opt_state, jnp.asarray(x_train[idx]), jnp.asarray(u_train[idx]))
            losses.append(float(loss))
        ll = float(val_ll(params, jnp.asarray(x_val), jnp.asarray(u_val)))
        history.append({"epoch": ep + 1, "train_nll": float(np.mean(losses)), "val_ll": ll})
        if ll > best[0]:
            best = (ll, params)
        if log is not None and (ep % 5 == 4 or ep == epochs - 1):
            log(f"    TDM epoch {ep + 1}: train NLL {np.mean(losses):.4f}, val LL {ll:.4f}")
    return model, best[1], history
