"""True per-beat propositions (labelling function L) computed from ``BeatStats``.

Thresholds are assumptions (HANDOFF sections 5.7 and 9; see docs/phase2_decisions.md):
    AVO   aortic valve opens: volume ejected through the aortic valve > eps_avo      (0.5 ml)
    SUC   suction: LVP reaches the suction threshold x1_bar (Rk > 0) at some instant
    BACK  backflow: volume regurgitated through the pump in the beat > back_ml      (1 ml)
          (back_ml = 0 is the strict definition "min Qp < 0 at some instant")
    LOW   low output: cardiac output < co_min                                         (4.4 l/min)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lvadrm.integrate import BeatStats

EPS_AVO_ML = 0.5
BACK_ML = 1.0
CO_MIN_LMIN = 4.4  # cardiac index 2.2 l/min/m2 with a body surface area of 2 m2


class Labels(NamedTuple):
    avo: jax.Array
    suc: jax.Array
    back: jax.Array
    low: jax.Array


def beat_labels(
    stats: BeatStats,
    x1_bar=1.0,
    eps_avo: float = EPS_AVO_ML,
    back_ml: float = BACK_ML,
    co_min: float = CO_MIN_LMIN,
) -> Labels:
    return Labels(
        avo=stats.v_aortic > eps_avo,
        suc=stats.lvp_min <= x1_bar,
        back=stats.v_back > back_ml,
        low=stats.co < co_min,
    )


def labels_array(labels: Labels):
    """Stack labels into a boolean array [..., 4] in the order (AVO, SUC, BACK, LOW)."""
    return jnp.stack(list(labels), axis=-1)
