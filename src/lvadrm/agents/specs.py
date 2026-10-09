"""What each agent observes, which reward it is trained on, and its action set.

oracle   pump signals + one-hot of the true RM state; RM reward               (H1 upper reference)
oracle_p pressures (sEDP, svalve, current and previous) + speed, HR, last actions + RM state;
         RM reward: the oracle with pressure sensors instead of flow features    (H3 reference)
rnn      pump signals only, GRU memory, no RM; RM reward                        (memory-only baseline)
mlp      pump signals only, no memory, no RM; RM reward                         (ablation)
fz       Fernandez-Zapico MDP: pressures (sEDP, svalve and their previous values), actions
         +-1 %, reward r_A + r_B                                                 (privileged replica)
fz4      same MDP with actions {-5, -1, +1, +5} %. With a zero action the FZ reward is degenerate
         (sEDP * 0 >= 0 always earns the maximum), so the 5-level set is replaced by 4 levels.

During training the true RM rewards are always available; the policy only sees ``agent_obs``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from lvadrm.env import EnvConfig, EnvState

FIVE_ACTIONS = (-0.05, -0.01, 0.0, 0.01, 0.05)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    obs_mode: str  # "pump" or "pressure"
    rm_in_obs: bool
    reward: str  # "rm" or "fz"
    actions: tuple[float, ...]
    recurrent: bool


SPECS = {
    "oracle": AgentSpec("oracle", "pump", True, "rm", FIVE_ACTIONS, False),
    "oracle_p": AgentSpec("oracle_p", "pressure_plus", True, "rm", FIVE_ACTIONS, False),
    "rnn": AgentSpec("rnn", "pump", False, "rm", FIVE_ACTIONS, True),
    "mlp": AgentSpec("mlp", "pump", False, "rm", FIVE_ACTIONS, False),
    "fz": AgentSpec("fz", "pressure", False, "fz", (0.01, -0.01), False),
    "fz4": AgentSpec("fz4", "pressure", False, "fz", (-0.05, -0.01, 0.01, 0.05), False),
}


def env_config_for(spec: AgentSpec, cfg: EnvConfig) -> EnvConfig:
    return replace(cfg, obs_mode=spec.obs_mode, actions=spec.actions)


def agent_obs(spec: AgentSpec, obs, state: EnvState, cfg: EnvConfig):
    """Policy input. Works for a single environment or a batch (leading axes)."""
    if spec.rm_in_obs:
        return jnp.concatenate([obs, jax.nn.one_hot(state.u, cfg.rm.n, dtype=obs.dtype)], axis=-1)
    return obs


def fz_reward(obs_prev, a_frac):
    """Fernandez-Zapico et al. 2024: r = r_A + r_B with r_A = 1 if sEDP * a >= 0 else -2 and
    r_B = 1 if svalve * a >= 0 else 0, from the observation the action was chosen on."""
    r_a = jnp.where(obs_prev[..., 0] * a_frac >= 0, 1.0, -2.0)
    r_b = jnp.where(obs_prev[..., 1] * a_frac >= 0, 1.0, 0.0)
    return r_a + r_b


def agent_reward(spec: AgentSpec, r_env, obs_prev, action, cfg: EnvConfig):
    if spec.reward == "rm":
        return r_env
    return fz_reward(obs_prev, jnp.asarray(cfg.actions, obs_prev.dtype)[action])
