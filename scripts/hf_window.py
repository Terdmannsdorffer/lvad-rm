"""Operating window of the HF patient at constant speed (float64, CPU).

For one-at-a-time variations of Rs, Emax, Vtot and HR around the base patient, finds the speed
at which the aortic valve stops opening (omega_avc), at which suction starts (omega_suc), and
below which the pump regurgitates more than ``back_ml`` per beat (omega_back), for several
BACK thresholds. Also reports the regurgitated volume per beat at omega_avc, i.e. the backflow
one has to accept to keep the valve opening at constant speed.

Writes results/phase2/hf_window.json and results/phase2/hf_window.png.
"""

from __future__ import annotations

import common

common.setup_cpu_x64()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lvadrm.patients import HF_PATIENT as P  # noqa: E402
from lvadrm.window import default_predicates, thresholds  # noqa: E402

GRID = np.arange(4000.0, 16000.0 + 1, 250.0)
BACK_ML = [0.0, 0.5, 1.0, 2.0, 5.0]
VARIATIONS = {
    "rs": [0.5, 0.8, 1.0, 1.5, 2.0],
    "emax_scale": [0.5, 0.75, 1.0, 1.25, 1.5],
    "vtot_scale": [0.85, 0.9, 1.0, 1.1],
    "hr": [60.0, 80.0, 100.0, 135.0],
}


def conditions():
    rows = []
    for name, values in VARIATIONS.items():
        for v in values:
            row = {"rs": P.rs, "emax": P.emax, "vtot": P.vtot, "hr": P.hr, "var": name, "value": v}
            if name == "rs":
                row["rs"] = v
            elif name == "emax_scale":
                row["emax"] = v * P.emax
            elif name == "vtot_scale":
                row["vtot"] = v * P.vtot
            else:
                row["hr"] = v
            rows.append(row)
    return rows


def main() -> None:
    rows = conditions()
    n_c = len(rows)
    cond = {
        "rs": np.array([r["rs"] for r in rows]),
        "emax": np.array([r["emax"] for r in rows]),
        "hr": np.array([r["hr"] for r in rows]),
        "emin": np.full(n_c, P.emin),
        "v0": np.full(n_c, P.v0),
    }
    vtot = np.array([r["vtot"] for r in rows])
    preds = default_predicates()
    preds.pop("omega_back")
    for b in BACK_ML:
        preds[f"omega_back_{b}ml"] = default_predicates(back_ml=b)["omega_back"]
    th, gs = thresholds(cond, vtot, GRID, preds, n_seg=64, n_inner=105)

    out = {"patient": P.__dict__, "back_ml": BACK_ML, "rows": []}
    for c, r in enumerate(rows):
        row = {"var": r["var"], "value": r["value"]} | {k: (None if np.isnan(v[c]) else float(v[c])) for k, v in th.items()}
        w_avc = th["omega_avc"][c]
        row["v_back_at_avc_ml"] = float(np.interp(w_avc, GRID, gs.v_back[c])) if np.isfinite(w_avc) else None
        row["co_at_avc_lmin"] = float(np.interp(w_avc, GRID, gs.co[c])) if np.isfinite(w_avc) else None
        out["rows"].append(row)
        print(
            f"{r['var']:10s} {r['value']:6.2f}: avc {row['omega_avc'] or float('nan'):7.0f}  suc {row['omega_suc'] or float('nan'):7.0f}  "
            + "  ".join(f"back{b} {row[f'omega_back_{b}ml'] or float('nan'):7.0f}" for b in BACK_ML)
            + f"  | v_back@avc {row['v_back_at_avc_ml'] or float('nan'):5.2f} ml, CO@avc {row['co_at_avc_lmin'] or float('nan'):.2f}"
        )
    out["grid"] = {"speeds": GRID, "v_back": gs.v_back, "v_aortic": gs.v_aortic, "co": gs.co, "lvp_min": gs.lvp_min}
    common.save_json("hf_window.json", out, phase=2)

    fig, axes = plt.subplots(1, len(VARIATIONS), figsize=(18, 4), sharey=True)
    for ax, name in zip(axes, VARIATIONS):
        idx = [c for c, r in enumerate(rows) if r["var"] == name]
        xs = [rows[c]["value"] for c in idx]
        ax.plot(xs, th["omega_suc"][idx] / 1000, "r.-", label="suction")
        ax.plot(xs, th["omega_avc"][idx] / 1000, "g.-", label="AV closes")
        for b, ls in zip(BACK_ML, (":", "--", "-.", "-", "-")):
            ax.plot(xs, th[f"omega_back_{b}ml"][idx] / 1000, "b" + ls, lw=0.9, label=f"back > {b} ml")
        ax.set(xlabel=name, title=f"HF patient: {name}")
    axes[0].set_ylabel("speed (krpm)")
    axes[0].legend(fontsize=7)
    common.savefig(fig, "hf_window.png", phase=2)
    plt.close(fig)
    print("wrote results/phase2/hf_window.json")


if __name__ == "__main__":
    main()
