"""V6, V6b, V6c: Simaan's minimum-pump-flow controller (Simaan et al. 2009, Section VI), float64 CPU.

Heart: Simaan's LVAD simulations as calibrated in Phase 1 (HR 60, Emax 2, Emin 0.06, V0 10,
Vtot 287.3 ml). The controller is the per-beat equivalent of the paper's 0.005 s updates
(staircase ramps; see lvadrm/baselines/simaan_minflow.py). Rs changes are applied at beat
boundaries.

V6   Rs = 1: speed rises from ~13.6 krpm and settles at ~15.25 krpm, below suction (Fig. 16)
V6b  Rs = 1 -> 2 between 27.5 and 32.5 s: ~15.25 -> ~14.25 krpm, always below suction (Fig. 17)
V6c  CO and MAP at Rs 0.5 / 1 / 2 with this controller and with the constant-Delta-p controller
     (Table V; expected to disagree, see docs/phase1_decisions.md)

Writes results/phase2/validate_controller.json and figures.
"""

from __future__ import annotations

import common

common.setup_cpu_x64()

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lvadrm.baselines import simaan_minflow as smf  # noqa: E402
from lvadrm.calibration import run_beats  # noqa: E402
from lvadrm.integrate import SAMPLE_NAMES, simulate_beat  # noqa: E402
from lvadrm.elastance import elastance  # noqa: E402
from lvadrm.model import AOP, LVV, initial_state, pump_static_flow  # noqa: E402
from lvadrm.model import QP as QP_STATE  # noqa: E402
from lvadrm.params import make_params  # noqa: E402
from lvadrm.patients import SIMAAN_PUMP_HEART as H  # noqa: E402

N_SEG, N_INNER = 256, 40  # N_sub = 10240 (validation grade), 256 samples per beat for plots
QP = SAMPLE_NAMES.index("qp")
OMEGA0 = 13600.0
SUCTION = {0.5: 17150.0, 1.0: 15450.0, 2.0: 14450.0}  # our Fig. 15 values (Phase 1, ramp_suction.py)
TABLE_V = {
    "minflow": {0.5: (14.4, 136.0), 1.0: (7.5, 133.0), 2.0: (3.8, 132.0)},
    "deltap": {0.5: (11.7, 110.0), 1.0: (6.5, 120.0), 2.0: (3.7, 137.0)},
}


def params(rs):
    return make_params(hr=H.hr, emax=H.emax, emin=H.emin, v0=H.v0, rs=rs)


OMEGA_LIMITS = (6000.0, 18000.0)


def run(rs_of_t, t_end, controller="minflow", omega0=OMEGA0, y0=None, c=smf.C_GAIN_FIG16, slew=None):
    """Closed-loop run, one beat per iteration. Returns per-beat arrays and the dense Qp trace."""
    tc = 60.0 / H.hr
    if y0 is None:
        # default end-diastolic guess (not periodic, as the start of Fig. 16b), with Qp at the pump's
        # static flow so that the first cycle minimum is not an artificial zero
        p0 = params(rs_of_t(0.0))
        y = initial_state(p0, H.vtot)
        lvp0 = float(elastance(0.0, p0.heart)) * (float(y[LVV]) - H.v0)
        y = y.at[QP_STATE].set(pump_static_flow(lvp0, float(y[AOP]), omega0, p0.pump))
    else:
        y = y0
    cs = smf.init(jnp.float64)
    omega, t, rate = omega0, 0.0, 0.0
    rec = {k: [] for k in ("t", "omega0", "omega_mean", "qp_min", "co", "map", "lvp_min", "rs")}
    qp_trace = []
    while t < t_end - 1e-9:
        rs = rs_of_t(t)
        y, s, smp = simulate_beat(y, params(rs), omega, rate, 0.0, smf.DT_UPDATE, n_seg=N_SEG, n_inner=N_INNER)
        for k, v in (("t", t), ("omega0", omega), ("omega_mean", s.omega_mean), ("qp_min", s.qp_min), ("co", s.co),
                     ("map", s.aop_mean), ("lvp_min", s.lvp_min), ("rs", rs)):
            rec[k].append(float(v))
        qp_trace.append(np.asarray(smp[:, QP]))
        omega = float(np.clip(omega + rate * tc, *OMEGA_LIMITS))  # staircase continues until the next minimum
        t += tc
        if controller == "minflow":
            cs = smf.push_minimum(cs, t - tc + s.t_qp_min, s.qp_min)
            rate = float(smf.speed_command(cs, t, c=c, slew=slew))
        elif controller == "deltap":
            rate = float(smf.deltap_rate(s.aop_mean, s.lvp_mean))
        # keep the ramp inside the speed limits
        rate = float(np.clip(rate, (OMEGA_LIMITS[0] - omega) / tc, (OMEGA_LIMITS[1] - omega) / tc))
    out = {k: np.array(v) for k, v in rec.items()}
    out["qp_trace"] = np.concatenate(qp_trace)
    out["t_trace"] = (np.arange(len(out["qp_trace"])) + 1) * tc / N_SEG
    return out


def rs_sim2(t):
    return 1.0 if t <= 27.5 else (1.0 + (t - 27.5) / 5.0 if t <= 32.5 else 2.0)


def plot(r, suction_line, name, title, r_literal=None, r_slew=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    tc = 60.0 / H.hr
    ax[0].plot(r["t"] + 0.5 * tc, r["omega_mean"] / 1000, lw=1, label=f"c = {smf.C_GAIN_FIG16} (K = 30)")
    if r_slew is not None:
        ax[0].plot(r_slew["t"] + 0.5 * tc, r_slew["omega_mean"] / 1000, lw=1, label=f"c = {smf.C_GAIN_FIG16}, slew {smf.SLEW_RPM_S:.0f} rpm/s")
    if r_literal is not None:
        ax[0].plot(r_literal["t"] + 0.5 * tc, r_literal["omega_mean"] / 1000, lw=0.8, alpha=0.6,
                   label=f"c = {smf.C_GAIN} as published (K = 152)")
    ax[0].plot(r["t"], suction_line(r["t"]) / 1000, "k--", lw=0.8, label="suction speed (our Fig. 15)")
    ax[0].set(xlabel="time (s)", ylabel="pump speed (krpm)", title=title)
    ax[0].legend(fontsize=8)
    ax[1].plot(r["t_trace"], r["qp_trace"], lw=0.5)
    ax[1].set(xlabel="time (s)", ylabel="pump flow (ml/s)")
    common.savefig(fig, name, phase=2)
    plt.close(fig)


def main() -> None:
    res = {}
    # V6: Rs = 1, start from the default initial guess at 13.6 krpm (not periodic, as in Fig. 16b)
    r1 = run(lambda t: 1.0, 60.0)
    r1_lit = run(lambda t: 1.0, 60.0, c=smf.C_GAIN)
    r1_slew = run(lambda t: 1.0, 60.0, slew=smf.SLEW_RPM_S)
    res["v6_slew"] = {
        "c": smf.C_GAIN_FIG16,
        "slew_rpm_s": smf.SLEW_RPM_S,
        "omega_at_10s": float(np.interp(10.0, r1_slew["t"], r1_slew["omega_mean"])),
        "omega_at_20s": float(np.interp(20.0, r1_slew["t"], r1_slew["omega_mean"])),
        "omega_final": r1_slew["omega_mean"][-1],
        "beats_with_suction": int(np.sum(r1_slew["lvp_min"] <= 1.0)),
    }
    res["v6_literal"] = {
        "c": smf.C_GAIN,
        "max_omega": r1_lit["omega_mean"].max(),
        "omega_final": r1_lit["omega_mean"][-1],
        "beats_with_suction": int(np.sum(r1_lit["lvp_min"] <= 1.0)),
    }
    res["v6"] = {
        "c": smf.C_GAIN_FIG16,
        "omega_at_10s": float(np.interp(10.0, r1["t"], r1["omega_mean"])),
        "omega_at_20s": float(np.interp(20.0, r1["t"], r1["omega_mean"])),
        "omega_start": r1["omega_mean"][0],
        "omega_final": r1["omega_mean"][-1],
        "max_omega": r1["omega_mean"].max(),
        "suction_speed": SUCTION[1.0],
        "beats_with_suction": int(np.sum(r1["lvp_min"] <= 1.0)),
        "target": "rises from ~13.6 krpm, settles ~15.25 krpm (< 15.5)",
    }
    plot(r1, lambda t: np.full_like(t, SUCTION[1.0]), "v6_minflow_rs1.png", "V6: Rs = 1 (Simaan Fig. 16)", r1_lit, r1_slew)

    # V6b: Rs ramp 1 -> 2 (eq. 15)
    r2 = run(rs_sim2, 60.0)
    r2_lit = run(rs_sim2, 60.0, c=smf.C_GAIN)
    r2_slew = run(rs_sim2, 60.0, slew=smf.SLEW_RPM_S)
    res["v6b_slew"] = {
        "omega_at_27s": float(np.interp(27.0, r2_slew["t"], r2_slew["omega_mean"])),
        "omega_final": r2_slew["omega_mean"][-1],
        "min_omega_after_ramp": float(r2_slew["omega_mean"][r2_slew["t"] > 32.5].min()),
        "beats_with_suction": int(np.sum(r2_slew["lvp_min"] <= 1.0)),
    }
    res["v6b"] = {
        "c": smf.C_GAIN_FIG16,
        "omega_at_27s": float(np.interp(27.0, r2["t"], r2["omega_mean"])),
        "omega_final": r2["omega_mean"][-1],
        "min_omega_after_ramp": float(r2["omega_mean"][r2["t"] > 32.5].min()),
        "beats_with_suction": int(np.sum(r2["lvp_min"] <= 1.0)),
        "target": "~15.25 -> ~14.25 krpm, always below suction (15 500 -> 14 400)",
    }
    plot(r2, lambda t: np.where(t < 30.0, SUCTION[1.0], SUCTION[2.0]), "v6b_minflow_rs_ramp.png", "V6b: Rs 1 -> 2 (Simaan Fig. 17)", r2_lit, r2_slew)

    # V6c: periodic regime of both controllers at three afterloads
    res["v6c"] = {}
    for rs in (0.5, 1.0, 2.0):
        row = {}
        for ctrl in ("minflow", "deltap"):
            r = run(lambda t, rs=rs: rs, 120.0, controller=ctrl, slew=smf.SLEW_RPM_S if ctrl == "minflow" else None)
            last = slice(-10, None)
            row[ctrl] = {
                "omega": float(r["omega_mean"][last].mean()),
                "co": float(r["co"][last].mean()),
                "map": float(r["map"][last].mean()),
                "table_v": TABLE_V[ctrl][rs],
                "suction_beats_last_60s": int(np.sum(r["lvp_min"][-60:] <= 1.0)),
            }
        res["v6c"][str(rs)] = row
        print(f"V6c Rs={rs}:", {k: (round(v['omega']), round(v['co'], 2), round(v['map'], 1), v['table_v']) for k, v in row.items()})

    # reference: periodic regime at the suction speed found by the ramp, Rs = 1 (CO at the edge of suction)
    _, st, _ = run_beats(params(1.0), H.vtot, SUCTION[1.0] - 200.0, n_beats=30, n_seg=64, n_inner=160)
    res["co_at_suction_minus_200rpm_rs1"] = float(np.asarray(st.co)[-1])

    for k in ("v6_literal", "v6", "v6_slew", "v6b", "v6b_slew"):
        print(k, res[k])
    common.save_json("validate_controller.json", res, phase=2)
    print("wrote results/phase2/validate_controller.json")


if __name__ == "__main__":
    main()
