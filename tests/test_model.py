"""Model tests (float64, CPU): conservation, equivalent forms, elastance, ESPVR, pump sign."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lvadrm.calibration import run_beats
from lvadrm.elastance import elastance, t_max
from lvadrm.integrate import simulate_beat
from lvadrm.labels import beat_labels
from lvadrm.model import G, LVV, QP, pump_flow_derivative
from lvadrm.params import HeartParams, make_params
from lvadrm.reference import reference_beat, stats_from_trajectory

VTOT = 287.0
SIM = {"n_seg": 64, "n_inner": 80}


def last(stats):
    return jax.tree.map(lambda x: np.asarray(x)[..., -1], stats)


@pytest.mark.parametrize("pump_on,omega", [(False, 0.0), (True, 13000.0), (True, 16000.0)])
def test_total_volume_is_conserved(pump_on, omega):
    p = make_params(hr=60.0)
    _, stats, _ = run_beats(p, VTOT, omega, n_beats=20, pump_on=pump_on, **SIM)
    assert np.max(np.abs(np.asarray(stats.vtot) - VTOT)) < 1e-9


def test_elastance_peak():
    heart = HeartParams(emax=2.0, emin=0.06, hr=60.0)
    t = jnp.linspace(0.0, 1.0, 100001)
    e = np.asarray(elastance(t, heart))
    i = int(np.argmax(e))
    assert abs(e[i] - 2.0) < 0.01
    assert abs(float(t[i]) - float(t_max(60.0))) < 0.005
    assert abs(e[-1] - 0.06) < 1e-6  # back to Emin at the end of the cycle


@pytest.mark.parametrize("pump_on,omega", [(False, 0.0), (True, 14000.0)])
def test_volume_and_pressure_forms_agree(pump_on, omega):
    p = make_params(hr=75.0)
    y0, _, _ = run_beats(p, VTOT, omega, n_beats=10, pump_on=pump_on, **SIM)
    y0 = np.asarray(y0)
    z0 = y0.copy()
    z0[LVV] = float(elastance(0.0, p.heart)) * (y0[LVV] - float(p.heart.v0))
    kw = {"pump_on": pump_on, "n_eval": 2001, "rtol": 1e-10, "atol": 1e-10, "max_step": 5e-4}
    rv = reference_beat(y0, p, omega, **kw)
    rp = reference_beat(z0, p, omega, pressure_form=True, **kw)
    np.testing.assert_allclose(rv["g"][:, G["lvp"]], rp["y"][:, LVV], atol=1e-6)
    np.testing.assert_allclose(rv["y"][:, QP], rp["y"][:, QP], atol=1e-6)


def test_espvr_afterload_slope():
    """End-systolic points (aortic valve closure) for two afterloads lie on a line of slope ~Emax."""
    p = make_params(hr=75.0, emin=0.05, rs=jnp.array([0.5, 2.0]))
    _, stats, _ = run_beats(p, jnp.array([270.0, 330.0]), n_beats=20, pump_on=False, **SIM)
    s = last(stats)
    slope = (s.esp[1] - s.esp[0]) / (s.esv[1] - s.esv[0])
    intercept = s.esv[0] - s.esp[0] / slope
    assert abs(slope - 2.0) < 0.05
    assert 9.0 < intercept < 13.0


def test_pump_sign_convention():
    q = make_params().pump
    lvp, aop = jnp.float64(5.0), jnp.float64(80.0)
    # consistent form: at zero flow the head drives flow forward (LV -> aorta) at 12 krpm
    assert pump_flow_derivative(lvp, aop, 0.0, 12000.0, q, "consistent") > 0
    # literal form: same situation drives flow backwards, and its branch resistance is negative
    assert pump_flow_derivative(lvp, aop, 0.0, 12000.0, q, "literal") < 0
    assert q.ri + q.ro + q.beta0 < 0
    # with the consistent form the steady pump flow is forward
    _, stats, _ = run_beats(make_params(hr=60.0), VTOT, 12000.0, n_beats=15, **SIM)
    assert float(last(stats).qp_mean) > 0


def test_on_the_fly_stats_match_reference():
    """RK4 per-beat statistics at a fine step agree with a Radau trajectory."""
    p = make_params(hr=60.0)
    y0, _, _ = run_beats(p, VTOT, 13000.0, n_beats=15, n_seg=64, n_inner=320)
    _, s, _ = simulate_beat(y0, p, 13000.0, n_seg=64, n_inner=320)
    ref = stats_from_trajectory(reference_beat(np.asarray(y0), p, 13000.0, n_eval=20481), p, 13000.0)
    for k in ("edv", "lvp_max", "aop_mean", "co", "qp_min", "qp_max", "qp_mean", "lap_mean", "vtot"):
        assert abs(float(getattr(s, k)) - ref[k]) <= 1e-3 * max(abs(ref[k]), 1.0), k


def test_labels():
    p = make_params(hr=60.0, rs=jnp.array([1.0, 1.0, 1.0]))
    _, stats, _ = run_beats(p, VTOT, jnp.array([7000.0, 10000.0, 16500.0]), n_beats=15, **SIM)
    lab = beat_labels(last(stats))
    avo, suc, back = np.asarray(lab.avo), np.asarray(lab.suc), np.asarray(lab.back)
    assert avo.tolist() == [True, True, False]
    assert suc.tolist() == [False, False, True]
    assert back[0] and not back[2]
