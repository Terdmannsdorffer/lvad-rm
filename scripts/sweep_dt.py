"""RK4 step-size sweep against a SciPy Radau reference (float64, CPU).

For each test case:
  * one-beat error: from the same periodic initial state, compare per-beat statistics of one
    RK4 beat at step dt with a Radau beat (rtol = atol = 1e-9);
  * steady-state error: run the RK4 model to its periodic regime at step dt and compare the
    last beat with the periodic regime at a very fine step (dt ~ 1.2e-5, itself checked
    against Radau by the one-beat test).
Errors are relative with floors (1 ml, 1 mmHg, 1 ml/s, 0.1 l/min) so that values near zero
do not blow up. Acceptance: max error < 1 % over the listed variables (HANDOFF 5.6).
Writes results/phase1/sweep_dt.json and a figure.
"""

from __future__ import annotations

import common

common.setup_cpu_x64()

import math  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lvadrm.calibration import run_beats  # noqa: E402
from lvadrm.integrate import simulate_beat  # noqa: E402
from lvadrm.params import make_params  # noqa: E402
from lvadrm.reference import reference_beat, stats_from_trajectory  # noqa: E402

DTS = [2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4]
EMAX_MAX = 2.0  # largest Emax in the scenarios (HANDOFF section 8)
FINE_N_INNER = 1280  # 64 * 1280 = 81920 substeps per beat (dt = 1.2e-5 at HR 60)
N_BEATS = 30

FLOORS = {
    "edv": 1.0, "esv": 1.0, "lvv_min": 1.0, "edp": 1.0, "lvp_max": 1.0, "lvp_min": 1.0,
    "aop_max": 1.0, "aop_min": 1.0, "aop_mean": 1.0, "ap_mean": 1.0, "lap_mean": 1.0,
    "co": 0.1, "qlv": 0.1, "v_aortic": 1.0, "qp_min": 1.0, "qp_max": 1.0, "qp_mean": 1.0,
}

CASES = {
    "no_pump_hr75": {"p": {"hr": 75.0}, "omega": 0.0, "pump_on": False},
    "pump_9k_avo_hr60": {"p": {}, "omega": 9000.0, "pump_on": True},
    "pump_13k_hr60": {"p": {}, "omega": 13000.0, "pump_on": True},
    "pump_15.8k_suction_hr60": {"p": {}, "omega": 15800.0, "pump_on": True},
    "pump_7k_backflow_rs2_hr60": {"p": {"rs": 2.0}, "omega": 7000.0, "pump_on": True},
    "pump_11k_emax0.6_hr60": {"p": {"emax": 0.6}, "omega": 11000.0, "pump_on": True},
    "pump_14k_exercise_rs0.6_hr135": {"p": {"rs": 0.6, "hr": 135.0}, "omega": 14000.0, "pump_on": True},
}


def n_inner_for(dt: float, hr: float) -> int:
    return max(1, math.ceil(60.0 / hr / dt / common.N_SEG))


def to_dict(stats) -> dict:
    return {k: float(np.asarray(v)) for k, v in stats._asdict().items()}


def rel_errors(a: dict, ref: dict) -> dict:
    out = {}
    for k, floor in FLOORS.items():
        if np.isnan(ref[k]) and np.isnan(a[k]):
            continue
        out[k] = abs(a[k] - ref[k]) / max(abs(ref[k]), floor)
    return out


def main() -> None:
    vtot = common.pump_vtot()
    results = {}
    for name, case in CASES.items():
        p = make_params(emin=common.PUMP_EMIN, **({"hr": common.PUMP_HR} | case["p"]))
        hr = float(p.heart.hr)
        w, pump_on = case["omega"], case["pump_on"]

        # periodic regime at a very fine step: initial state for the one-beat test and steady-state reference
        y_fine, st_fine, _ = run_beats(p, vtot, w, n_beats=N_BEATS, pump_on=pump_on, n_seg=common.N_SEG, n_inner=FINE_N_INNER)
        ss_ref = to_dict(jax.tree.map(lambda x: x[-1], st_fine))

        ref = reference_beat(np.asarray(y_fine), p, w, pump_on=pump_on, rtol=1e-9, atol=1e-9, max_step=5e-4, n_eval=40001)
        beat_ref = stats_from_trajectory(ref, p, w)

        res = {
            "hr": hr,
            "omega": w,
            "params": case["p"],
            "pump_on": pump_on,
            "radau_nfev": int(ref["nfev"]),
            "steady_reference": ss_ref,
            "one_beat": {},
            "steady": {},
        }
        fine_err = rel_errors(
            to_dict(simulate_beat(y_fine, p, w, n_seg=common.N_SEG, n_inner=FINE_N_INNER, pump_on=pump_on)[1]), beat_ref
        )
        res["fine_vs_radau_max_err"] = max(fine_err.values())
        for dt in DTS:
            n_inner = n_inner_for(dt, hr)
            dt_eff = 60.0 / hr / (common.N_SEG * n_inner)
            _, sb, _ = simulate_beat(y_fine, p, w, n_seg=common.N_SEG, n_inner=n_inner, pump_on=pump_on)
            e1 = rel_errors(to_dict(sb), beat_ref)
            _, st, _ = run_beats(p, vtot, w, n_beats=N_BEATS, pump_on=pump_on, n_seg=common.N_SEG, n_inner=n_inner)
            s_last = to_dict(jax.tree.map(lambda x: x[-1], st))
            finite = bool(np.all(np.isfinite([v for k, v in s_last.items() if k not in ("esv", "esp")])))
            e2 = rel_errors(s_last, ss_ref) if finite else {"nonfinite": np.inf}
            res["one_beat"][str(dt)] = {"dt_eff": dt_eff, "max_err": max(e1.values()), "worst": max(e1, key=e1.get), "errors": e1}
            res["steady"][str(dt)] = {"dt_eff": dt_eff, "max_err": max(e2.values()), "worst": max(e2, key=e2.get), "errors": e2}
            print(f"{name:32s} dt={dt:.1e} (eff {dt_eff:.2e}): one-beat max {100 * max(e1.values()):.4f}% ({max(e1, key=e1.get)}), "
                  f"steady max {100 * max(e2.values()):.4f}% ({max(e2, key=e2.get)})")
        print(f"{name:32s} fine-step vs Radau: {100 * res['fine_vs_radau_max_err']:.5f}%  (Radau nfev {ref['nfev']})")
        results[name] = res

    worst = {
        str(dt): max(max(r["one_beat"][str(dt)]["max_err"], r["steady"][str(dt)]["max_err"]) for r in results.values())
        for dt in DTS
    }
    # The stiffest mode is ejection (aortic valve open): tau = Ra * (C_LV series Ca) = Ra Ca / (1 + Ca E),
    # and explicit RK4 is stable for dt < 2.785 tau. The ideal diode can mask an unstable step
    # (the valve closes and clips the oscillation), so we also require a 20 % margin to that limit.
    c = make_params().circ
    tau = c.ra * c.ca / (1.0 + c.ca * EMAX_MAX)
    dt_stab = 2.785 * tau
    ok = [dt for dt in DTS if worst[str(dt)] < 0.01 and dt <= 0.8 * dt_stab]
    recommended = max(ok) if ok else None
    print("worst error per dt:", {k: f"{100 * v:.3f}%" for k, v in worst.items()})
    print(f"RK4 stability limit (aortic valve open, Emax {EMAX_MAX}): {dt_stab:.3e} s -> recommended dt: {recommended}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for name, r in results.items():
        ax[0].loglog(DTS, [max(r["one_beat"][str(d)]["max_err"], 1e-12) * 100 for d in DTS], "o-", label=name)
        ax[1].loglog(DTS, [min(max(r["steady"][str(d)]["max_err"], 1e-12), 1e3) * 100 for d in DTS], "o-", label=name)
    for a, title in zip(ax, ("one beat vs Radau", "periodic regime vs fine step")):
        a.axhline(1.0, c="k", ls="--", lw=0.8)
        a.set(xlabel="dt (s) at the case's HR", title=title)
    ax[0].set_ylabel("max relative error over per-beat variables (%)")
    ax[1].legend(fontsize=6)
    common.savefig(fig, "sweep_dt.png")
    plt.close(fig)

    common.save_json(
        "sweep_dt.json",
        {"vtot": vtot, "cases": results, "worst_per_dt": worst, "dt_stability_limit": dt_stab,
         "recommended_dt": recommended, "dts": DTS},
    )
    print("wrote results/phase1/sweep_dt.json")


if __name__ == "__main__":
    main()
