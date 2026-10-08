"""Phase 1 validation of the unassisted heart and the model equations (float64, CPU).

V1  elastance peak (Simaan Fig. 3)
V2  healthy heart without pump, HR 75; calibrates Vtot (Simaan Fig. 5)
V3  ESPVR under afterload (Rs) and preload (Rm) changes (Simaan Fig. 6)
V9  volume conservation and convergence to a periodic regime
--  volume form vs the paper's pressure form
--  the literal pump equations of the paper (eqs. 11-13) are unstable / pump backwards

Writes results/phase1/validate_simaan.json and figures.
"""

from __future__ import annotations

import common

common.setup_cpu_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from lvadrm.calibration import HEALTHY_TARGETS, calibrate_healthy_vtot, run_beats  # noqa: E402
from lvadrm.elastance import elastance, t_max  # noqa: E402
from lvadrm.integrate import SAMPLE_NAMES, simulate_beats  # noqa: E402
from lvadrm.model import G, LVV, QP, rhs, total_volume  # noqa: E402
from lvadrm.params import HeartParams, make_params  # noqa: E402
from lvadrm.reference import reference_beat  # noqa: E402

SIM = {"n_seg": common.N_SEG, "n_inner": common.N_INNER}
S = {name: i for i, name in enumerate(SAMPLE_NAMES)}


def last(stats):
    return jax.tree.map(lambda x: np.asarray(x)[..., -1], stats)


def trajectory(y0, p, omega, n_beats, pump_on):
    """Dense samples (every substep) over ``n_beats`` beats starting from ``y0``."""
    n_sub = common.N_SEG * common.N_INNER
    w = jnp.full(n_beats, omega, dtype=y0.dtype)
    _, stats, samples = simulate_beats(y0, p, w, n_seg=n_sub, n_inner=1, pump_on=pump_on)
    tc = 60.0 / float(p.heart.hr)
    t = (np.arange(n_beats * n_sub) + 1) * tc / n_sub
    return t, np.asarray(samples).reshape(n_beats * n_sub, -1), stats


# --------------------------------------------------------------------------------------- V1
def v1_elastance() -> dict:
    heart = HeartParams(emax=2.0, emin=0.06, hr=60.0)
    t = np.linspace(0.0, 1.0, 200001)
    e = np.asarray(elastance(jnp.asarray(t), heart))
    i = int(np.argmax(e))
    de = np.asarray(jax.vmap(jax.grad(lambda tt: elastance(tt, heart)))(jnp.asarray(t)))

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(t, e)
    ax[0].axvline(float(t_max(60.0)), ls="--", c="gray", lw=0.8)
    ax[0].set(xlabel="time (s)", ylabel="E(t) (mmHg/ml)", title="Elastance, HR 60 (Simaan Fig. 3)")
    ax[1].plot(t, de / e)
    ax[1].set(xlabel="time (s)", ylabel="-C'(t)/C(t) = E'/E (1/s)", title="Simaan Fig. 4")
    common.savefig(fig, "v1_elastance.png")
    plt.close(fig)
    return {
        "peak_E": float(e[i]),
        "t_peak": float(t[i]),
        "Tmax": float(t_max(60.0)),
        "E_end_of_cycle": float(e[-1]),
        "Edot_over_E_range": [float((de / e).min()), float((de / e).max())],
    }


# --------------------------------------------------------------------------------------- V2
def v2_healthy() -> dict:
    out = {"targets": HEALTHY_TARGETS, "fits": {}}
    for emin in (0.05, 0.06):
        vtot, m = calibrate_healthy_vtot(emin=emin, **SIM)
        p = make_params(hr=75.0, emax=2.0, emin=emin, v0=10.0, rs=1.0)
        y, stats, _ = run_beats(p, vtot, n_beats=40, pump_on=False, **SIM)
        s = last(stats)
        m.update(
            vtot=vtot,
            map_x3=float(s.ap_mean),
            edv=float(s.edv),
            esv=float(s.esv),
            lap_mean=float(s.lap_mean),
            rel_err={k: (m[k] - v) / v for k, v in HEALTHY_TARGETS.items()},
        )
        out["fits"][str(emin)] = m
    best = min(out["fits"], key=lambda k: out["fits"][k]["cost"])
    out["chosen_emin"] = float(best)
    out["chosen_vtot"] = out["fits"][best]["vtot"]

    # waveform figure (Simaan Fig. 5)
    p = make_params(hr=75.0, emax=2.0, emin=float(best), v0=10.0, rs=1.0)
    y, _, _ = run_beats(p, out["chosen_vtot"], n_beats=40, pump_on=False, **SIM)
    t, smp, _ = trajectory(y, p, 0.0, 4, pump_on=False)
    fig, ax = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    ax[0].plot(t, smp[:, S["aop"]], label="AoP")
    ax[0].plot(t, smp[:, S["lvp"]], label="LVP", alpha=0.8)
    ax[0].plot(t, smp[:, S["lap"]], label="LAP")
    ax[0].set(ylabel="Pressure (mmHg)")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(t, smp[:, S["qa"]])
    ax[1].set(ylabel="Aortic flow (ml/s)")
    ax[2].plot(t, smp[:, S["lvv"]])
    ax[2].set(ylabel="LVV (ml)", xlabel="time (s)")
    fig.suptitle(f"Healthy heart, no pump, HR 75 (Simaan Fig. 5); Vtot = {out['chosen_vtot']:.1f} ml")
    common.savefig(fig, "v2_healthy_waveforms.png")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------- V3
def _vtot_for_edv(p, edv_target, lo=150.0, hi=600.0, n_grid=25, n_rounds=3):
    """Per-condition Vtot giving the target EDV (EDV increases with Vtot). ``p`` is batched (n,)."""
    n = len(np.atleast_1d(np.asarray(p.circ.rs)))
    lo = np.full(n, lo)
    hi = np.full(n, hi)
    for _ in range(n_rounds):
        grid = np.linspace(lo, hi, n_grid)  # [n_grid, n]
        rep = jax.tree.map(lambda a: jnp.repeat(jnp.atleast_1d(a)[None], n_grid, 0).reshape(-1) if jnp.ndim(a) else a, p)
        _, stats, _ = run_beats(rep, jnp.asarray(grid.reshape(-1)), n_beats=40, pump_on=False, **SIM)
        edv = last(stats).edv.reshape(n_grid, n)
        vt = np.array([np.interp(edv_target, edv[:, j], grid[:, j]) for j in range(n)])
        step = grid[1] - grid[0]
        lo, hi = vt - step, vt + step
    return vt


def _espvr_fit(esv, esp):
    slope, b = np.polyfit(esv, esp, 1)
    return float(slope), float(-b / slope)


def v3_espvr(vtot_cal: float) -> dict:
    base = {"hr": 75.0, "emax": 2.0, "emin": 0.05, "v0": 10.0}
    out = {"setup": base | {"note": "Simaan Section III: Emax 2, Emin 0.05, V0 10"}}

    # afterload: Rs varies, EDV held at the Rs = 1 value (Simaan Fig. 6a keeps EDV constant)
    rs = np.array([0.5, 1.0, 1.5, 2.0])
    p_ref = make_params(rs=1.0, **base)
    _, st_ref, _ = run_beats(p_ref, vtot_cal, n_beats=40, pump_on=False, **SIM)
    edv_ref = float(last(st_ref).edv)
    p_aft = make_params(rs=jnp.asarray(rs), **base)
    vt_aft = _vtot_for_edv(p_aft, edv_ref)
    y_aft, st_aft, _ = run_beats(p_aft, jnp.asarray(vt_aft), n_beats=40, pump_on=False, **SIM)
    s_aft = last(st_aft)

    # preload: Rm varies at fixed Vtot, Rs = 1
    rm = np.array([0.005, 0.08, 0.32, 0.8])
    p_pre = make_params(rs=1.0, rm=jnp.asarray(rm), **base)
    y_pre, st_pre, _ = run_beats(p_pre, vtot_cal, n_beats=40, pump_on=False, **SIM)
    s_pre = last(st_pre)

    out["afterload"] = {
        "rs": rs,
        "vtot": vt_aft,
        "edv": s_aft.edv,
        "esv": s_aft.esv,
        "esp": s_aft.esp,
        "fit_avc": _espvr_fit(s_aft.esv, s_aft.esp),
        "target": [1.98, 10.91],
    }
    out["preload"] = {
        "rm": rm,
        "vtot": vtot_cal,
        "edv": s_pre.edv,
        "esv": s_pre.esv,
        "esp": s_pre.esp,
        "fit_avc": _espvr_fit(s_pre.esv, s_pre.esp),
        "target": [1.92, 8.84],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, ys, labels in (
        (axes[0], "afterload", y_aft, [f"Rs = {r}" for r in rs]),
        (axes[1], "preload", y_pre, [f"Rm = {r}" for r in rm]),
    ):
        pk = p_aft if key == "afterload" else p_pre
        for j, lab in enumerate(labels):
            pj = jax.tree.map(lambda a: a[j] if jnp.ndim(a) else a, pk)
            _, smp, _ = trajectory(ys[j], pj, 0.0, 1, pump_on=False)
            ax.plot(smp[:, S["lvv"]], smp[:, S["lvp"]], lw=1, label=lab)
        d = out[key]
        m, vint = d["fit_avc"]
        vv = np.linspace(vint, max(d["esv"]) + 15, 10)
        ax.plot(vv, m * (vv - vint), "k-", lw=1)
        ax.plot(d["esv"], d["esp"], "ko", ms=4, mfc="none")
        ax.set_xlabel("LVV (ml)")
        ax.set_title(
            f"{key}: ESPVR slope {m:.2f} mmHg/ml, intercept {vint:.2f} ml\n"
            f"(Simaan: {d['target'][0]} and {d['target'][1]})",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 180)
    axes[0].set_ylabel("LVP (mmHg)")
    common.savefig(fig, "v3_pv_loops_espvr.png")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------- V9
def _beats_to_periodic(stats, tol_edv=1e-3, tol_map=1e-3):
    edv = np.asarray(stats.edv)
    aop = np.asarray(stats.aop_mean)
    ok = (np.abs(np.diff(edv)) < tol_edv) & (np.abs(np.diff(aop)) < tol_map)
    # first beat index after which all subsequent changes stay below tolerance
    bad = np.nonzero(~ok)[0]
    return int(bad[-1] + 2) if len(bad) else 1


def v9_conservation(vtot_cal: float, emin: float) -> dict:
    out = {}
    cases = {
        "no_pump_hr75": (make_params(hr=75.0, emin=emin), 0.0, False),
        "pump_14krpm_hr60": (make_params(hr=common.PUMP_HR, emin=emin), 14000.0, True),
        "pump_15.5krpm_suction_hr60": (make_params(hr=common.PUMP_HR, emin=emin), 15500.0, True),
    }
    for name, (p, w, pump_on) in cases.items():
        _, stats, _ = run_beats(p, vtot_cal, w, n_beats=200, pump_on=pump_on, **SIM)
        vt = np.asarray(stats.vtot)
        out[name] = {
            "vtot_max_abs_drift_ml": float(np.max(np.abs(vt - vtot_cal))),
            "vtot_rel_drift": float(np.max(np.abs(vt - vtot_cal)) / vtot_cal),
            "beats_to_periodic(dEDV<1e-3 ml, dMAP<1e-3 mmHg)": _beats_to_periodic(stats),
        }
    return out


# ---------------------------------------------------------------------- pressure vs volume form
def forms_check(vtot_cal: float, emin: float) -> dict:
    out = {}
    cases = {
        "no_pump_hr75": (make_params(hr=75.0, emin=emin), 0.0, False),
        "pump_14krpm_hr60": (make_params(hr=common.PUMP_HR, emin=emin), 14000.0, True),
    }
    for name, (p, w, pump_on) in cases.items():
        y0, _, _ = run_beats(p, vtot_cal, w, n_beats=30, pump_on=pump_on, **SIM)
        y0 = np.asarray(y0)
        z0 = y0.copy()
        z0[LVV] = float(elastance(0.0, p.heart)) * (y0[LVV] - float(p.heart.v0))
        kw = {"pump_on": pump_on, "n_eval": 5001, "rtol": 1e-10, "atol": 1e-10, "max_step": 2e-4}
        rv = reference_beat(y0, p, w, **kw)
        rp = reference_beat(z0, p, w, pressure_form=True, **kw)
        lvp_v = rv["g"][:, G["lvp"]]
        lvp_p = rp["y"][:, LVV]
        out[name] = {
            "max_abs_diff_lvp_mmHg": float(np.max(np.abs(lvp_v - lvp_p))),
            "lvp_range_mmHg": float(lvp_v.max() - lvp_v.min()),
            "max_abs_diff_qp_mlps": float(np.max(np.abs(rv["y"][:, QP] - rp["y"][:, QP]))),
        }
    return out


# ---------------------------------------------------------------------------- literal pump form
def literal_pump(vtot_cal: float, emin: float) -> dict:
    p = make_params(hr=common.PUMP_HR, emin=emin)
    q = p.pump
    out = {
        "literal": {"L_star": q.li + q.lo + q.beta1, "R_star_no_suction": q.ri + q.ro + q.beta0},
        "consistent": {"L_star": q.li + q.lo - q.beta1, "R_star_no_suction": q.ri + q.ro - q.beta0},
    }
    for k in out:
        out[k]["branch_pole_1_per_s"] = -out[k]["R_star_no_suction"] / out[k]["L_star"]

    w = 12000.0
    y0, _, _ = run_beats(p, vtot_cal, w, n_beats=30, pump_on=True, **SIM)
    y0 = np.asarray(y0)
    tc = 60.0 / common.PUMP_HR

    def make_fun(form, omega):
        f = jax.jit(lambda t, y: rhs(jnp.mod(t, tc), y, omega, p, True, form)[0])
        return lambda t, y: np.asarray(f(t, y))

    def blowup(t, y):
        return np.abs(y[QP]) - 1e4

    blowup.terminal = True
    runs = {}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for form, omega in (("consistent", w), ("literal", w), ("literal", 0.0)):
        sol = solve_ivp(make_fun(form, omega), (0.0, 5.0), y0, method="LSODA", rtol=1e-8, atol=1e-8,
                        max_step=1e-3, events=blowup)
        key = f"{form}_omega{int(omega)}"
        runs[key] = {
            "t_end": float(sol.t[-1]),
            "blew_up(|Qp|>1e4 ml/s)": bool(len(sol.t_events[0])),
            "qp_min": float(sol.y[QP].min()),
            "qp_max": float(sol.y[QP].max()),
            "vtot_end": float(total_volume(sol.y[:, -1], p)),
        }
        for a in ax:
            a.plot(sol.t, sol.y[QP], label=key)
    ax[0].set(xlabel="time (s)", ylabel="Qp (ml/s)", title="Pump flow from a 12 krpm steady state", yscale="symlog")
    ax[1].set(xlabel="time (s)", ylabel="Qp (ml/s)", xlim=(0, 0.3), ylim=(-2000, 400), title="zoom")
    ax[0].legend(fontsize=8)
    common.savefig(fig, "literal_pump_form.png")
    plt.close(fig)
    out["runs"] = runs
    return out


def main() -> None:
    res = {"v1": v1_elastance()}
    print("V1", res["v1"])
    res["v2"] = v2_healthy()
    vtot, emin = res["v2"]["chosen_vtot"], res["v2"]["chosen_emin"]
    print("V2", {k: {kk: vv for kk, vv in v.items() if kk != "rel_err"} for k, v in res["v2"]["fits"].items()})
    res["v3"] = v3_espvr(res["v2"]["fits"]["0.05"]["vtot"])
    print("V3 afterload", res["v3"]["afterload"]["fit_avc"], "preload", res["v3"]["preload"]["fit_avc"])
    res["v9"] = v9_conservation(vtot, emin)
    print("V9", res["v9"])
    res["forms"] = forms_check(vtot, emin)
    print("forms", res["forms"])
    res["literal_pump"] = literal_pump(vtot, emin)
    print("literal pump", res["literal_pump"])
    print("wrote", common.save_json("validate_simaan.json", res))


if __name__ == "__main__":
    main()
