"""Double-hill time-varying elastance of the left ventricle (Simaan et al. 2009, eqs. 2-3)."""

from __future__ import annotations

import jax.numpy as jnp

from lvadrm.params import HeartParams


def cycle_length(hr):
    """Cardiac cycle length tc = 60 / HR [s]."""
    return 60.0 / hr


def t_max(hr):
    """Time to peak elastance Tmax = 0.2 + 0.15 tc [s]."""
    return 0.2 + 0.15 * cycle_length(hr)


def normalized_elastance(tn):
    """En(tn) = 1.55 * [(tn/0.7)^1.9 / (1 + (tn/0.7)^1.9)] * [1 / (1 + (tn/1.17)^21.9)]."""
    a = (tn / 0.7) ** 1.9
    b = (tn / 1.17) ** 21.9
    return 1.55 * (a / (1.0 + a)) / (1.0 + b)


def elastance(t, heart: HeartParams):
    """E(t) [mmHg/ml] at beat-local time t in [0, tc] (no modulo: the integrators work beat by beat)."""
    tn = t / t_max(heart.hr)
    return (heart.emax - heart.emin) * normalized_elastance(tn) + heart.emin


def elastance_periodic(t, heart: HeartParams):
    """E(t) for global time t, using mod(t, tc)."""
    return elastance(jnp.mod(t, cycle_length(heart.hr)), heart)
