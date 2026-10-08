"""Model parameters.

Cardiovascular values from Simaan et al. (2009), Table I; LVAD values from Table IV
(axial pump model of Choi et al. 1997 for the Nimbus / HeartMate II).

All parameter containers are frozen dataclasses registered as JAX pytrees, so any
field can be a scalar or an array (e.g. one value per environment under ``vmap``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HeartParams:
    """Left ventricle (time-varying elastance) and heart rate."""

    emax: float = 2.0  # end-systolic elastance [mmHg/ml]
    emin: float = 0.05  # diastolic elastance [mmHg/ml]
    v0: float = 10.0  # unstressed volume [ml]
    hr: float = 60.0  # heart rate [bpm]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CircParams:
    """Valves, left atrium / pulmonary compliance and 4-element Windkessel afterload."""

    rs: float = 1.0  # systemic vascular resistance (SVR) [mmHg s/ml]
    rm: float = 0.0050  # mitral valve resistance [mmHg s/ml]
    ra: float = 0.0010  # aortic valve resistance [mmHg s/ml]
    rc: float = 0.0398  # characteristic resistance [mmHg s/ml]
    cr: float = 4.4000  # left atrial compliance [ml/mmHg]
    cs: float = 1.3300  # systemic compliance [ml/mmHg]
    ca: float = 0.0800  # aortic compliance [ml/mmHg]
    ls: float = 0.0005  # aortic blood inertance [mmHg s^2/ml]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PumpParams:
    """Rotary LVAD: cannulae, suction resistance and pump head curve.

    The head curve H = beta0*Qp + beta1*dQp/dt + beta2*omega^2 is the pressure *rise*
    (outlet - inlet) for these coefficients (omega in rpm); see HANDOFF.md section 5.4.
    """

    ri: float = 0.0677  # inlet cannula resistance [mmHg s/ml]
    ro: float = 0.0677  # outlet cannula resistance [mmHg s/ml]
    li: float = 0.0127  # inlet cannula inertance [mmHg s^2/ml]
    lo: float = 0.0127  # outlet cannula inertance [mmHg s^2/ml]
    alpha: float = -3.5  # suction resistance slope [s/ml]
    x1_bar: float = 1.0  # suction threshold on LVP [mmHg]
    beta0: float = -0.17070  # [mmHg s/ml]
    beta1: float = -0.02177  # [mmHg s^2/ml]
    beta2: float = 9.9025e-7  # [mmHg/rpm^2]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Params:
    heart: HeartParams = field(default_factory=HeartParams)
    circ: CircParams = field(default_factory=CircParams)
    pump: PumpParams = field(default_factory=PumpParams)


def make_params(**overrides: float) -> Params:
    """Build ``Params`` from flat keyword overrides, e.g. ``make_params(rs=2.0, hr=75.0)``."""
    groups = {"heart": HeartParams, "circ": CircParams, "pump": PumpParams}
    kwargs: dict[str, dict[str, float]] = {name: {} for name in groups}
    for key, value in overrides.items():
        for name, cls in groups.items():
            if key in cls.__dataclass_fields__:
                kwargs[name][key] = value
                break
        else:
            raise KeyError(f"unknown parameter {key!r}")
    return Params(**{name: cls(**kwargs[name]) for name, cls in groups.items()})
