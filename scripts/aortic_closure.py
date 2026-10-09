"""V8: steady-state speed at which the aortic valve stops opening, vs Rs and vs Emax.

Also reports, on the same steady-state runs, the speed below which there is backflow
through the pump (strict definition: min Qp < 0) and the speed at which suction starts
(LVP <= x1_bar), so together they give the operating window [omega_back, omega_suc] and where
AVO is lost.

Heart: Emax 2 (unless varied), Emin 0.06, V0 10, HR 60, Vtot calibrated (common.py).
Vtot is kept fixed when Emax changes (no baroreflex, no fluid retention): an assumption.
Thresholds come from a 250 rpm grid refined by bisection. Writes results/phase1/aortic_closure.json.
"""

from __future__ import annotations

import common

common.setup_cpu_x64()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lvadrm.labels import EPS_AVO_ML  # noqa: E402
from lvadrm.window import default_predicates, thresholds  # noqa: E402

GRID = np.arange(4000.0, 18000.0 + 1, 250.0)
RS_LIST = [0.5, 1.0, 1.5, 2.0, 2.5]
EMAX_LIST = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0]


def main() -> None:
    vtot = common.pump_vtot()
    rs = np.array(RS_LIST + [1.0] * len(EMAX_LIST))
    emax = np.array([2.0] * len(RS_LIST) + EMAX_LIST)
    n_c = len(rs)
    cond = {"rs": rs, "emax": emax, "hr": np.full(n_c, common.PUMP_HR), "emin": np.full(n_c, common.PUMP_EMIN)}
    th, grid_stats = thresholds(cond, np.full(n_c, vtot), GRID, default_predicates(), n_seg=common.N_SEG, n_inner=common.N_INNER)

    out = {"vtot": vtot, "hr": common.PUMP_HR, "emin": common.PUMP_EMIN, "eps_avo_ml": EPS_AVO_ML, "vs_rs": {}, "vs_emax": {}}
    for c in range(n_c):
        row = {k: (None if np.isnan(v[c]) else float(v[c])) for k, v in th.items()}
        (out["vs_rs"] if c < len(RS_LIST) else out["vs_emax"])[str(rs[c] if c < len(RS_LIST) else emax[c])] = row
        print(f"Rs={rs[c]} Emax={emax[c]}:", {k: round(float(v[c])) if not np.isnan(v[c]) else None for k, v in th.items()})

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for j, (sel, xs, xlabel, key) in enumerate(
        (
            (slice(0, len(RS_LIST)), RS_LIST, "Rs (mmHg s/ml), Emax = 2", "rs"),
            (slice(len(RS_LIST), n_c), EMAX_LIST, "Emax (mmHg/ml), Rs = 1", "emax"),
        )
    ):
        for name, style in (("omega_back", "b.-"), ("omega_avc", "g.-"), ("omega_suc", "r.-")):
            ax[0, j].plot(xs, th[name][sel] / 1000, style, label=name)
        ax[0, j].set(xlabel=xlabel, ylabel="speed (krpm)", title="AVO lost above omega_avc; window [back, suc]")
        ax[0, j].legend(fontsize=8)
        for c, x in zip(range(n_c)[sel], xs):
            ax[1, j].plot(GRID / 1000, grid_stats.qlv[c], label=f"{key} = {x}")
        ax[1, j].set(xlabel="speed (krpm)", ylabel="mean aortic valve flow QLV (l/min)")
        ax[1, j].legend(fontsize=7)
    common.savefig(fig, "v8_aortic_closure.png")
    plt.close(fig)

    out["grid"] = {
        "speeds": GRID,
        "rs": rs,
        "emax": emax,
        "qlv_lmin": grid_stats.qlv,
        "co_lmin": grid_stats.co,
        "aop_mean": grid_stats.aop_mean,
        "qp_min": grid_stats.qp_min,
    }
    common.save_json("aortic_closure.json", out)
    print("wrote results/phase1/aortic_closure.json")


if __name__ == "__main__":
    main()
