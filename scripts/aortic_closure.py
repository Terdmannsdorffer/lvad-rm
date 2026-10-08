"""V8: steady-state speed at which the aortic valve stops opening, vs Rs and vs Emax.

Also reports, on the same steady-state runs, the speed below which there is backflow
through the pump (min Qp < 0) and the speed at which suction starts (LVP <= x1_bar), so
together they give the operating window [omega_back, omega_suc] and where AVO is lost.

Heart: Emax 2 (unless varied), Emin 0.06, V0 10, HR 60, Vtot calibrated (common.py).
Vtot is kept fixed when Emax changes (no baroreflex, no fluid retention): an assumption.
Thresholds come from a 250 rpm grid refined by bisection. Writes results/phase1/aortic_closure.json.
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

from lvadrm.calibration import run_beats  # noqa: E402
from lvadrm.labels import EPS_AVO_ML  # noqa: E402
from lvadrm.params import make_params  # noqa: E402

N_BEATS = 30
GRID = np.arange(4000.0, 18000.0 + 1, 250.0)
N_BISECT = 9  # 250 rpm / 2^9 < 0.5 rpm

RS_LIST = [0.5, 1.0, 1.5, 2.0, 2.5]
EMAX_LIST = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0]

# predicate(stats_last) is False below the threshold and True above it
PREDICATES = {
    "omega_avc": lambda s: s.v_aortic <= EPS_AVO_ML,  # aortic valve no longer opens
    "omega_back": lambda s: s.qp_min >= 0.0,  # no more backflow through the pump
    "omega_suc": lambda s: s.lvp_min <= 1.0,  # suction resistance switches on
}


def conditions():
    rs = np.array(RS_LIST + [1.0] * len(EMAX_LIST))
    emax = np.array([2.0] * len(RS_LIST) + EMAX_LIST)
    return rs, emax


def steady_last(rs, emax, omega, vtot):
    p = make_params(hr=common.PUMP_HR, emin=common.PUMP_EMIN, rs=jnp.asarray(rs), emax=jnp.asarray(emax))
    _, stats, _ = run_beats(p, vtot, jnp.asarray(omega), n_beats=N_BEATS, n_seg=common.N_SEG, n_inner=common.N_INNER)
    return jax.tree.map(lambda x: np.asarray(x)[..., -1], stats)


def main() -> None:
    vtot = common.pump_vtot()
    rs, emax = conditions()
    n_c, n_g = len(rs), len(GRID)

    # grid: all conditions x all speeds in one batch
    s = steady_last(np.repeat(rs, n_g), np.repeat(emax, n_g), np.tile(GRID, n_c), vtot)
    grid_stats = jax.tree.map(lambda x: x.reshape(n_c, n_g), s)

    thresholds = {}
    for name, pred in PREDICATES.items():
        flags = pred(grid_stats)  # [n_c, n_g]
        lo = np.full(n_c, np.nan)
        hi = np.full(n_c, np.nan)
        for c in range(n_c):
            idx = np.nonzero(flags[c])[0]
            if len(idx) and idx[0] > 0 and np.all(flags[c, idx[0]:]):
                lo[c], hi[c] = GRID[idx[0] - 1], GRID[idx[0]]
            elif len(idx) and idx[0] == 0:
                hi[c] = GRID[0]  # already true at the lowest speed
        ok = ~np.isnan(lo)
        for _ in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            sm = steady_last(rs[ok], emax[ok], mid[ok], vtot)
            f = pred(sm)
            lo[ok] = np.where(f, lo[ok], mid[ok])
            hi[ok] = np.where(f, mid[ok], hi[ok])
        thresholds[name] = np.where(ok, 0.5 * (lo + hi), np.nan)

    out = {"vtot": vtot, "hr": common.PUMP_HR, "emin": common.PUMP_EMIN, "eps_avo_ml": EPS_AVO_ML, "vs_rs": {}, "vs_emax": {}}
    for c in range(n_c):
        row = {k: (None if np.isnan(v[c]) else float(v[c])) for k, v in thresholds.items()}
        if c < len(RS_LIST):
            out["vs_rs"][str(rs[c])] = row
        else:
            out["vs_emax"][str(emax[c])] = row
    for c in range(n_c):
        print(f"Rs={rs[c]} Emax={emax[c]}:", {k: round(float(v[c])) if not np.isnan(v[c]) else None for k, v in thresholds.items()})

    # figure: thresholds and aortic-valve flow vs speed
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for j, (sel, xs, xlabel, key) in enumerate(
        (
            (slice(0, len(RS_LIST)), RS_LIST, "Rs (mmHg s/ml), Emax = 2", "rs"),
            (slice(len(RS_LIST), n_c), EMAX_LIST, "Emax (mmHg/ml), Rs = 1", "emax"),
        )
    ):
        for name, style in (("omega_back", "b.-"), ("omega_avc", "g.-"), ("omega_suc", "r.-")):
            ax[0, j].plot(xs, thresholds[name][sel] / 1000, style, label=name)
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
