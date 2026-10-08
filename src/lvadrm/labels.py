"""True per-beat propositions (labelling function L) computed from ``BeatStats``.

Thresholds are assumptions to be confirmed (HANDOFF sections 5.7 and 9):
    AVO   aortic valve opens: volume through the aortic valve > eps_avo        (default 0.5 ml)
    SUC   suction: LVP reaches the suction threshold x1_bar (Rk > 0) at some instant
    BACK  backflow through the pump: min Qp < 0
    LOW   low perfusion: cardiac output < co_min                               (default 5.0 l/min)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lvadrm.integrate import BeatStats

EPS_AVO_ML = 0.5
CO_MIN_LMIN = 5.0


class Labels(NamedTuple):
    avo: jax.Array
    suc: jax.Array
    back: jax.Array
    low: jax.Array


def beat_labels(stats: BeatStats, x1_bar=1.0, eps_avo: float = EPS_AVO_ML, co_min: float = CO_MIN_LMIN) -> Labels:
    return Labels(
        avo=stats.v_aortic > eps_avo,
        suc=stats.lvp_min <= x1_bar,
        back=stats.qp_min < 0.0,
        low=stats.co < co_min,
    )


def labels_array(labels: Labels):
    """Stack labels into a boolean array [..., 4] in the order (AVO, SUC, BACK, LOW)."""
    return jnp.stack(list(labels), axis=-1)
