"""V4 and V5: open-loop speed ramps 12 -> 18 krpm in 60 s (Simaan Figs. 12, 14, 15).

The suction speed is the mean speed of the beat at which the envelope of the per-beat
minimum pump flow peaks (resolution: one beat, i.e. 100 rpm at HR 60). We also report the
speed at which the suction resistance first switches on (LVP <= x1_bar, label SUC) and the
speed at which backflow (min Qp < 0) disappears.

The heart (Emin with its own calibrated Vtot, HR) is chosen as the combination that best
reproduces Fig. 15. Writes results/phase1/ramp_suction.json and figures.
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

from lvadrm.calibration import batch_axes, calibrate_healthy_vtot  # noqa: E402
from lvadrm.integrate import SAMPLE_NAMES, simulate_beats  # noqa: E402
from lvadrm.model import initial_state  # noqa: E402
from lvadrm.params import make_params  # noqa: E402

FIG15 = {0.5: 17400.0, 1.0: 15500.0, 1.5: 14800.0, 2.0: 14400.0, 2.5: 14100.0}  # Simaan Figs. 14-15
W_START, W_END, T_RAMP = 12000.0, 18000.0, 60.0
N_SETTLE = 30
N_SEG_PLOT, N_INNER_PLOT = 256, 40  # same N_sub as common.N_SEG * common.N_INNER
assert N_SEG_PLOT * N_INNER_PLOT == common.N_SEG * common.N_INNER


def ramp_schedule(hr: float):
    tc = 60.0 / hr
    n_ramp = int(round(T_RAMP / tc))
    rate = (W_END - W_START) / T_RAMP
    w0 = np.concatenate([np.full(N_SETTLE, W_START), W_START + rate * tc * np.arange(n_ramp)])
    wr = np.concatenate([np.zeros(N_SETTLE), np.full(n_ramp, rate)])
    return w0, wr


def run_ramp(emin: float, vtot: float, hr: float, rs: np.ndarray):
    p = make_params(hr=hr, emax=2.0, emin=emin, v0=10.0, rs=jnp.asarray(rs))
    w0, wr = ramp_schedule(hr)

    def one(pp):
        y0 = initial_state(pp, vtot)
        return simulate_beats(y0, pp, jnp.asarray(w0), jnp.asarray(wr), n_seg=N_SEG_PLOT, n_inner=N_INNER_PLOT)

    _, stats, samples = jax.vmap(one, in_axes=(batch_axes(p),))(p)
    stats = jax.tree.map(lambda x: np.asarray(x)[:, N_SETTLE:], stats)
    return stats, np.asarray(samples)[:, N_SETTLE:]


def summarize(stats, rs: np.ndarray) -> dict:
    out = {}
    for i, r in enumerate(rs):
        w = stats.omega_mean[i]
        qmin = stats.qp_min[i]
        k = int(np.argmax(qmin))
        suc = np.nonzero(stats.lvp_min[i] <= 1.0)[0]
        back = np.nonzero(qmin < 0.0)[0]
        out[str(r)] = {
            "suction_speed_envelope_peak": float(w[k]),
            "target": FIG15[float(r)],
            "rel_err": float((w[k] - FIG15[float(r)]) / FIG15[float(r)]),
            "qp_min_at_peak": float(qmin[k]),
            "co_at_peak_lmin": float(stats.co[i][k]),
            "aop_mean_at_peak": float(stats.aop_mean[i][k]),
            "first_speed_with_Rk_on": float(w[suc[0]]) if len(suc) else None,
            "last_speed_with_backflow": float(w[back[-1]]) if len(back) else None,
            "aortic_valve_opens_during_ramp": bool(np.any(stats.v_aortic[i] > 0.5)),
        }
    return out


def main() -> None:
    rs = np.array(list(FIG15))
    combos = {}
    for emin in (0.05, 0.06):
        vtot, _ = calibrate_healthy_vtot(emin=emin, n_seg=common.N_SEG, n_inner=common.N_INNER)
        for hr in (60.0, 75.0):
            stats, samples = run_ramp(emin, vtot, hr, rs)
            summ = summarize(stats, rs)
            errs = np.array([v["rel_err"] for v in summ.values()])
            key = f"emin{emin}_hr{int(hr)}"
            combos[key] = {
                "emin": emin,
                "vtot": vtot,
                "hr": hr,
                "rms_rel_err": float(np.sqrt(np.mean(errs**2))),
                "max_abs_rel_err": float(np.max(np.abs(errs))),
                "per_rs": summ,
            }
            combos[key]["_data"] = (stats, samples)
            print(key, f"vtot={vtot:.2f}", "rms", f"{combos[key]['rms_rel_err']:.4f}",
                  {r: v["suction_speed_envelope_peak"] for r, v in summ.items()})

    best = min(combos, key=lambda k: combos[k]["rms_rel_err"])
    # Simaan's pump simulations run at HR 60 (Fig. 16b); prefer it unless HR 75 is clearly better
    hr60 = min((k for k in combos if k.endswith("hr60")), key=lambda k: combos[k]["rms_rel_err"])
    chosen = hr60
    print("best overall:", best, "chosen (HR 60):", chosen)

    stats, samples = combos[chosen]["_data"]
    hr = combos[chosen]["hr"]
    tc = 60.0 / hr
    qp = samples[..., SAMPLE_NAMES.index("qp")].reshape(len(rs), -1)
    t = (np.arange(qp.shape[1]) + 1) * tc / N_SEG_PLOT
    w_t = W_START + (W_END - W_START) / T_RAMP * t

    # V4: Rs = 1 (Simaan Fig. 12)
    i1 = list(rs).index(1.0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(t, w_t / 1000)
    ax[0].set(xlabel="time (s)", ylabel="speed (krpm)", title="Ramp")
    ax[1].plot(t, qp[i1], lw=0.5)
    ax[1].plot((np.arange(len(stats.qp_min[i1])) + 0.5) * tc, stats.qp_min[i1], "r.", ms=3, label="min Qp per beat")
    ws = combos[chosen]["per_rs"]["1.0"]["suction_speed_envelope_peak"]
    ax[1].axvline((ws - W_START) / (W_END - W_START) * T_RAMP, c="k", ls="--", lw=0.8)
    ax[1].set(xlabel="time (s)", ylabel="pump flow (ml/s)", title=f"Rs = 1: suction at {ws:.0f} rpm (Simaan: 15 500)")
    ax[1].legend(fontsize=8)
    common.savefig(fig, "v4_ramp_rs1.png")
    plt.close(fig)

    # V5: all Rs (Simaan Fig. 14)
    fig, axes = plt.subplots(1, len(rs), figsize=(16, 3), sharey=True)
    for i, r in enumerate(rs):
        d = combos[chosen]["per_rs"][str(r)]
        axes[i].plot(t, qp[i], lw=0.4)
        axes[i].set(title=f"Rs = {r}: {d['suction_speed_envelope_peak']:.0f} rpm\n(Simaan {d['target']:.0f})", xlabel="time (s)")
    axes[0].set_ylabel("pump flow (ml/s)")
    common.savefig(fig, "v5_ramp_all_rs.png")
    plt.close(fig)

    # V5: suction speed vs Rs (Simaan Fig. 15)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(rs, [FIG15[float(r)] / 1000 for r in rs], "ko-", label="Simaan Fig. 15")
    for key, c in combos.items():
        ax.plot(rs, [c["per_rs"][str(r)]["suction_speed_envelope_peak"] / 1000 for r in rs], ".-",
                label=f"{key} (rms {100 * c['rms_rel_err']:.1f}%)", lw=2 if key == chosen else 0.8)
    ax.set(xlabel="Rs (mmHg s/ml)", ylabel="suction speed (krpm)")
    ax.legend(fontsize=7)
    common.savefig(fig, "v5_suction_speed_vs_rs.png")
    plt.close(fig)

    for c in combos.values():
        c.pop("_data")
    common.save_json("ramp_suction.json", {"combos": combos, "best_overall": best, "chosen": chosen})
    print("wrote results/phase1/ramp_suction.json")


if __name__ == "__main__":
    main()
