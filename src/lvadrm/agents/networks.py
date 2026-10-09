"""Actor-critic networks (Flax), PureJaxRL style."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class ActorCritic(nn.Module):
    """Separate two-layer tanh MLPs for the policy logits and the value."""

    n_actions: int
    hidden: int = 128

    @nn.compact
    def __call__(self, x):
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(a))
        logits = nn.Dense(self.n_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        c = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        c = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(c))
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return logits, jnp.squeeze(v, -1)


class ActorCriticRNN(nn.Module):
    """Observation encoder -> GRU -> actor and critic heads. One step: (h, x) -> (h', logits, v)."""

    n_actions: int
    hidden: int = 128

    @nn.compact
    def __call__(self, h, x):
        e = nn.relu(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        h, y = nn.GRUCell(features=self.hidden)(h, e)
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(y))
        logits = nn.Dense(self.n_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        c = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(y))
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return h, logits, jnp.squeeze(v, -1)

    def initial_state(self, batch_shape=()):
        return jnp.zeros((*batch_shape, self.hidden))
