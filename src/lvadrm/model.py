"""Right-hand side of the 0D LV + systemic circulation + rotary LVAD model (Simaan et al. 2009).

State vector (volume form; equivalent to the paper's pressure form, see ``rhs_pressure_form``):

    y = [lvv, lap, ap, aop, qt, qp]
        lvv  left ventricular volume              [ml]   (x1 = LVP = E(t) * (lvv - V0))
        lap  left atrial pressure, x2             [mmHg]
        ap   systemic arterial pressure, x3       [mmHg]
        aop  aortic pressure, x4                  [mmHg]
        qt   total flow through Rc and Ls, x5     [ml/s]
        qp   pump flow, x6                        [ml/s]

Besides dy/dt, ``rhs`` returns a vector of instantaneous quantities ``g`` (names in
``G_NAMES``). The integrators integrate ``g`` with the same RK4 weights as the state, so
per-beat means and volumes are consistent with the state update.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lvadrm.elastance import elastance
from lvadrm.params import Params, PumpParams

STATE_NAMES = ("lvv", "lap", "ap", "aop", "qt", "qp")
LVV, LAP, AP, AOP, QT, QP = range(len(STATE_NAMES))

G_NAMES = ("qa", "qm", "qp", "qt", "lap", "ap", "aop", "lvp", "suction", "qs")
G = {name: i for i, name in enumerate(G_NAMES)}

PUMP_FORMS = ("consistent", "literal")


def _relu(x):
    return jnp.maximum(x, 0.0)


def suction_resistance(lvp, pump: PumpParams):
    """Rk = alpha * (x1 - x1_bar) if x1 <= x1_bar else 0 (Simaan eq. 8; alpha < 0 so Rk >= 0)."""
    return pump.alpha * jnp.minimum(lvp - pump.x1_bar, 0.0)


def pump_flow_derivative(lvp, aop, qp, omega, pump: PumpParams, form: str = "consistent"):
    """dQp/dt of the pump branch (inlet cannula + suction resistor + pump + outlet cannula).

    ``consistent``: H = beta0*Qp + beta1*dQp/dt + beta2*omega^2 is the pressure rise, so
        (Li + Lo - beta1) dQp/dt = x1 - x4 - (Ri + Ro + Rk - beta0) Qp + beta2 omega^2.
    ``literal``: Simaan eqs. 11-13 as printed (L* = Li + Lo + beta1, R* = Ri + Ro + Rk + beta0,
        b6 = -beta2 / L*). Kept only to document that it is unstable / pumps backwards.
    """
    rk = suction_resistance(lvp, pump)
    if form == "consistent":
        l_star = pump.li + pump.lo - pump.beta1
        r_star = pump.ri + pump.ro + rk - pump.beta0
        return (lvp - aop - r_star * qp + pump.beta2 * omega**2) / l_star
    if form == "literal":
        l_star = pump.li + pump.lo + pump.beta1
        r_star = pump.ri + pump.ro + rk + pump.beta0
        return (lvp - aop - r_star * qp - pump.beta2 * omega**2) / l_star
    raise ValueError(f"unknown pump form {form!r}; expected one of {PUMP_FORMS}")


def rhs(t, y, omega, p: Params, pump_on: bool = True, pump_form: str = "consistent"):
    """Volume-form right-hand side. Returns (dy/dt, g) for a single state ``y`` of shape (6,).

    ``t`` is beat-local time in [0, tc]. With ``pump_on=False`` the pump branch is removed
    (Qp = 0); setting omega = 0 would instead turn the pump into a leak.
    """
    h, c = p.heart, p.circ
    lvv, lap, ap, aop, qt, qp = (y[i] for i in range(len(STATE_NAMES)))

    lvp = elastance(t, h) * (lvv - h.v0)
    qm = _relu(lap - lvp) / c.rm  # mitral valve (ideal diode + Rm)
    qa = _relu(lvp - aop) / c.ra  # aortic valve (ideal diode + Ra)
    qs = (ap - lap) / c.rs  # systemic flow through SVR back to the atrium (closed loop)

    if pump_on:
        dqp = pump_flow_derivative(lvp, aop, qp, omega, p.pump, pump_form)
        suction = (lvp <= p.pump.x1_bar).astype(y.dtype)
    else:
        qp = jnp.zeros_like(lvv)
        dqp = jnp.zeros_like(lvv)
        suction = jnp.zeros_like(lvv)

    dy = jnp.stack(
        [
            qm - qa - qp,
            (qs - qm) / c.cr,
            (qt - qs) / c.cs,
            (qa + qp - qt) / c.ca,
            (aop - ap - c.rc * qt) / c.ls,
            dqp,
        ]
    )
    g = jnp.stack([qa, qm, qp, qt, lap, ap, aop, lvp, suction, qs])
    return dy, g


def rhs_pressure_form(t, z, omega, p: Params, pump_on: bool = True, pump_form: str = "consistent"):
    """Paper's pressure form: state z = [x1, lap, ap, aop, qt, qp] with
    dx1/dt = -(C'/C) x1 + (Qm - Qa - Qp)/C = (E'/E) x1 + E (Qm - Qa - Qp).

    Used only to check numerically that it matches the volume form.
    """
    e, e_dot = jax.value_and_grad(lambda tt: elastance(tt, p.heart))(t)
    y = z.at[LVV].set(z[LVV] / e + p.heart.v0)
    dy, g = rhs(t, y, omega, p, pump_on, pump_form)
    dz = dy.at[LVV].set(e_dot / e * z[LVV] + e * dy[LVV])
    return dz, g


def total_volume(y, p: Params):
    """Vtot = LVV + Cr x2 + Cs x3 + Ca x4 [ml]; conserved exactly (inertances store no volume)."""
    c = p.circ
    return y[..., LVV] + c.cr * y[..., LAP] + c.cs * y[..., AP] + c.ca * y[..., AOP]


def initial_state(p: Params, vtot, lap=8.0, ap=95.0, qp=0.0):
    """End-diastolic guess with a given total volume: pressures set, LVV takes the remainder."""
    c = p.circ
    lap = jnp.asarray(lap, dtype=jnp.result_type(float))
    ap = jnp.asarray(ap, dtype=lap.dtype)
    lvv = vtot - c.cr * lap - (c.cs + c.ca) * ap
    zero = jnp.zeros_like(lvv)
    return jnp.stack([lvv, lap + zero, ap + zero, ap + zero, zero, qp + zero], axis=-1)
