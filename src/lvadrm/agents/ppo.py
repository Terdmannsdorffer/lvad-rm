"""PPO and recurrent PPO for the beat-by-beat LVAD environment (PureJaxRL style, adapted).

Episodes have a fixed length that is a multiple of the rollout length, so all environments are
reset together at the start of a rollout (no auto-reset). The last step of an episode is a
time-limit truncation and is bootstrapped with the value of the final observation.

    train_state = init(key)
    for _ in range(n_updates):
        train_state, metrics = update(train_state)       # one rollout + PPO epochs (jitted)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from lvadrm.agents.networks import ActorCritic, ActorCriticRNN
from lvadrm.agents.specs import AgentSpec, agent_obs, agent_reward
from lvadrm.env import EnvConfig, EnvState, reset, step


@dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 8192
    n_steps: int = 25  # rollout length; must divide the episode length
    total_beats: int = 30_000_000
    n_epochs: int = 4
    minibatch_size: int = 8192  # transitions per minibatch (recurrent: n_steps * envs per minibatch)
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128

    @property
    def n_updates(self) -> int:
        return self.total_beats // (self.n_envs * self.n_steps)

    @property
    def n_minibatches(self) -> int:
        return max(1, self.n_envs * self.n_steps // self.minibatch_size)

    @property
    def n_minibatches_recurrent(self) -> int:
        """Recurrent minibatches split environments (full sequences): the divisor of n_envs closest
        to the transition-based count."""
        divisors = [d for d in range(1, self.n_envs + 1) if self.n_envs % d == 0]
        return min(divisors, key=lambda d: (abs(d - self.n_minibatches), d))


class TrainState(NamedTuple):
    params: Any
    opt_state: Any
    env_state: EnvState
    obs: jax.Array
    h: jax.Array  # recurrent state [n_envs, hidden] (unused for the MLP)
    key: jax.Array
    update: jax.Array


class Transition(NamedTuple):
    x: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    info: dict


def gae(values, rewards, last_v, gamma: float, lam: float):
    """Generalized advantage estimation over [T, ...] without terminations (truncation is
    bootstrapped with ``last_v``). Returns (advantages, value targets)."""

    def body(carry, xs):
        adv_next, v_next = carry
        v, r = xs
        delta = r + gamma * v_next - v
        adv = delta + gamma * lam * adv_next
        return (adv, v), adv

    _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), (values, rewards), reverse=True)
    return adv, adv + values


def make_network(spec: AgentSpec, cfg: PPOConfig):
    if spec.recurrent:
        return ActorCriticRNN(n_actions=len(spec.actions), hidden=cfg.hidden)
    return ActorCritic(n_actions=len(spec.actions), hidden=cfg.hidden)


def make_ppo(spec: AgentSpec, env_cfg: EnvConfig, cfg: PPOConfig):
    """Returns (init, update) for this agent / environment / PPO configuration."""
    assert env_cfg.episode_beats % cfg.n_steps == 0, "rollout length must divide the episode length"
    net = make_network(spec, cfg)
    n = cfg.n_envs
    n_mb = cfg.n_minibatches_recurrent if spec.recurrent else cfg.n_minibatches
    total_grad_steps = cfg.n_updates * cfg.n_epochs * n_mb
    lr = optax.linear_schedule(cfg.lr, 0.0, total_grad_steps) if cfg.anneal_lr else cfg.lr
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5))
    reset_v = jax.vmap(reset, in_axes=(0, None))
    step_v = jax.vmap(step, in_axes=(0, 0, 0, None))

    def apply(params, h, x):
        if spec.recurrent:
            return net.apply(params, h, x)
        logits, v = net.apply(params, x)
        return h, logits, v

    def init(key) -> TrainState:
        k_net, k_env, k_run = jax.random.split(key, 3)
        obs, st = reset_v(jax.random.split(k_env, n), env_cfg)
        x = agent_obs(spec, obs, st, env_cfg)
        h = jnp.zeros((n, cfg.hidden), x.dtype)
        params = net.init(k_net, h[0], x[0]) if spec.recurrent else net.init(k_net, x[0])
        return TrainState(params, tx.init(params), st, obs, h, k_run, jnp.int32(0))

    def rollout(ts: TrainState):
        def env_step(carry, _):
            st, obs, h, key = carry
            key, k_a, k_s = jax.random.split(key, 3)
            x = agent_obs(spec, obs, st, env_cfg)
            h2, logits, v = apply(ts.params, h, x)
            a = jax.random.categorical(k_a, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[:, None], axis=-1)[:, 0]
            obs2, st2, r_env, _, info = step_v(jax.random.split(k_s, n), st, a, env_cfg)
            r = agent_reward(spec, r_env, obs, a, env_cfg)
            small = {
                "violation": info["violation"],
                "labels": info["labels"],
                "co": info["co"],
                "omega": info["omega"],
                "r_env": r_env,
            }
            return (st2, obs2, h2, key), Transition(x, a, logp, v, r, small)

        (st, obs, h, key), traj = jax.lax.scan(env_step, (ts.env_state, ts.obs, ts.h, ts.key), None, cfg.n_steps)
        _, _, last_v = apply(ts.params, h, agent_obs(spec, obs, st, env_cfg))
        return st, obs, h, key, traj, last_v

    def loss_fn(params, h0, batch, adv, targets):
        x, a, old_logp, old_v = batch
        if spec.recurrent:  # x: [T, B, d]
            def body(h, xt):
                h, logits, v = net.apply(params, h, xt)
                return h, (logits, v)

            _, (logits, v) = jax.lax.scan(body, h0, x)
        else:
            logits, v = net.apply(params, x)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, a[..., None], axis=-1)[..., 0]
        ratio = jnp.exp(logp - old_logp)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_n).mean()
        v_clip = old_v + jnp.clip(v - old_v, -cfg.clip_eps, cfg.clip_eps)
        vf = 0.5 * jnp.maximum((v - targets) ** 2, (v_clip - targets) ** 2).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        total = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
        aux = {
            "pg_loss": pg,
            "vf_loss": vf,
            "entropy": ent,
            "approx_kl": ((ratio - 1.0) - jnp.log(ratio)).mean(),
            "clip_frac": (jnp.abs(ratio - 1.0) > cfg.clip_eps).mean(),
        }
        return total, aux

    @jax.jit
    def update(ts: TrainState):
        # synchronized reset at episode boundaries
        key, k_reset = jax.random.split(ts.key)
        need_reset = ts.env_state.k[0] >= env_cfg.episode_beats
        obs, st, h = jax.lax.cond(
            need_reset,
            lambda: (*reset_v(jax.random.split(k_reset, n), env_cfg), jnp.zeros_like(ts.h)),
            lambda: (ts.obs, ts.env_state, ts.h),
        )
        h0 = h
        ts = ts._replace(env_state=st, obs=obs, h=h, key=key)
        st, obs, h, key, traj, last_v = rollout(ts)
        adv, targets = gae(traj.value, traj.reward, last_v, cfg.gamma, cfg.gae_lambda)

        def epoch(carry, k):
            params, opt_state = carry
            if spec.recurrent:  # minibatches over environments, full sequences
                perm = jax.random.permutation(k, n).reshape(n_mb, -1)

                def mb(carry, idx):
                    params, opt_state = carry
                    batch = (traj.x[:, idx], traj.action[:, idx], traj.logp[:, idx], traj.value[:, idx])
                    (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, h0[idx], batch, adv[:, idx], targets[:, idx])
                    upd, opt_state = tx.update(g, opt_state, params)
                    return (optax.apply_updates(params, upd), opt_state), aux
            else:  # minibatches over flattened transitions
                flat = jax.tree.map(lambda a: a.reshape(-1, *a.shape[2:]), (traj.x, traj.action, traj.logp, traj.value, adv, targets))
                perm = jax.random.permutation(k, n * cfg.n_steps).reshape(n_mb, -1)

                def mb(carry, idx):
                    params, opt_state = carry
                    x, a, lp, v, ad, tg = (f[idx] for f in flat)
                    (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, None, (x, a, lp, v), ad, tg)
                    upd, opt_state = tx.update(g, opt_state, params)
                    return (optax.apply_updates(params, upd), opt_state), aux

            return jax.lax.scan(mb, (params, opt_state), perm)

        key, k_ep = jax.random.split(key)
        (params, opt_state), aux = jax.lax.scan(epoch, (ts.params, ts.opt_state), jax.random.split(k_ep, cfg.n_epochs))
        lab = traj.info["labels"]
        metrics = {k: v.mean() for k, v in aux.items()} | {
            "reward": traj.reward.mean(),
            "r_env": traj.info["r_env"].mean(),
            "violations_per_100": 100.0 * traj.info["violation"].mean(),
            "avo": lab[..., 0].mean(),
            "suc": lab[..., 1].mean(),
            "back": lab[..., 2].mean(),
            "low": lab[..., 3].mean(),
            "co": traj.info["co"].mean(),
            "omega": traj.info["omega"].mean(),
            "value": traj.value.mean(),
        }
        new_ts = TrainState(params, opt_state, st, obs, h, key, ts.update + 1)
        return new_ts, metrics

    return init, update, net


def greedy_policy(spec: AgentSpec, env_cfg: EnvConfig, net, params, hidden: int, stochastic: bool = False):
    """Evaluation policy for ``lvadrm.evaluation.rollout`` (single environment, vmapped there)."""
    from lvadrm.evaluation import Policy

    acts = jnp.asarray(env_cfg.actions)

    def init(state, obs):
        return jnp.zeros((hidden,), obs.dtype)

    def act(key, st, obs, h):
        x = agent_obs(spec, obs, st, env_cfg)
        if spec.recurrent:
            h, logits, _ = net.apply(params, h, x)
        else:
            logits, _ = net.apply(params, x)
        a = jax.random.categorical(key, logits) if stochastic else jnp.argmax(logits)
        return st.omega * (1.0 + acts[a]), 0.0, 0.0, h

    def observe(h, st, info):
        return h

    return Policy(init, act, observe)
