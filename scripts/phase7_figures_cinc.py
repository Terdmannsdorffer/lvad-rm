"""Phase 7: figures of the four-page Computing in Cardiology paper, at their printed size and with 9 pt text as the conference
requires (CPU), and the numbers of the robustness experiments (docs/phase7_decisions.md, section 7).

    python scripts/phase7_figures_cinc.py
Writes results/phase7/figures/cinc{1,2}.pdf|png and results/phase7/robust/numbers.json. Panels without data stay empty.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from phase7_common import OUT
from phase7_figures import FIG, INK, INK2, STYLE, XVIOL, by_bias, front, load, paired
from phase7_robust import CAL_ERR, DONE, DRIFTS, LAMS, SHIFT_POLICIES, SHIFTS, key, lookup
from phase7_tables import episodes, pick, worst5

WIDTH = 171.75 / 25.4  # text width of cinc.cls [in]
RC = {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
      "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"}
AMBER = {20.0: "#d8b46a", 10.0: "#a77a1c", 0.0: "#5c4008"}  # darker = better calibrated
SHORT = {"constant": "Constant", "blind": "Blind pattern", "true": "True count", "naive": "Naive", "ibu_d": "IBU", "ibu_hist": "IBU recurrent", "tdm": "TDM"}


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(FIG / f"{name}.{ext}", **kw)
    plt.close(fig)
    print("wrote", name)


def relabel(ax):
    h, lab = ax.get_legend_handles_labels()
    inv = {v[3]: SHORT.get(k, v[3]) for k, v in STYLE.items()}
    return [(a, inv.get(b, b)) for a, b in zip(h, lab)]


def legend(fig, axes, ncol, y=0.0):
    seen = {}
    for ax in axes:
        for a, b in relabel(ax):
            seen.setdefault(b, a)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), columnspacing=1.0, handlelength=1.6, handletextpad=0.5,
               borderaxespad=0.1)


def viol_axis(ax, hi):
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.5)
    ax.set_xlim(-0.02, hi)
    ax.set_xticks([0, 0.1, 1, 10])
    ax.set_xticklabels(["0", "0.1", "1", "10"])


def fig1():
    """Counting pays, and so does feedback when the patient changes."""
    m, r = load("map.json"), load("monitor/reference_N10_ramp.json")
    b = m["named"]["base"]
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.25))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.91, bottom=0.31, wspace=0.36)
    sp, co, op = np.array(b["speeds"]) / 1000, np.array(b["co"]), np.array(b["open"])
    ax[0].plot(sp[op], co[op], color="#2a78d6", lw=2)
    ax[0].plot(sp[~op], co[~op], color="#8a877f", lw=2, ls="--")
    ax[0].axvline(b["w_avo"] / 1000, color=INK2, lw=0.8)
    ax[0].axhline(5.0, color=INK2, lw=0.8, ls=":")
    ax[0].text(6.1, 5.06, "target", color=INK2)
    ax[0].text(b["w_avo"] / 1000 - 0.1, 5.75, "opens in\nevery beat", ha="right", va="center")
    ax[0].text(b["w_avo"] / 1000 + 0.15, 3.55, "never opens", va="center")
    ax[0].set(xlabel="Constant speed [krpm]", ylabel="Output [l/min]", xlim=(6, 10.5), ylim=(3.2, 6.3), title="(a) Base patient")
    ns = [3, 5, 10, 20]
    gains = np.array([[p["reference"][str(n)]["gain_pct"] for n in ns] for p in m["named"].values()])
    ax[1].plot(range(4), gains.T, color="#b9b6ad", lw=0.9)
    ax[1].plot(range(4), gains[0], color="#2a78d6", lw=2, marker="o")
    for i, g in enumerate(gains[0]):
        ax[1].annotate(f"+{g:.0f}%", (i, g), textcoords="offset points", xytext=(-3, 5), ha="right")
    ax[1].set_xticks(range(4), [str(n) for n in ns])
    ax[1].set(xlabel="Beats allowed, N", ylabel="Output gain [%]", xlim=(-0.6, 3.2), ylim=(-2, 48), title="(b) Best pattern, 8 patients")
    for k in ("constant", "blind", "true"):
        front(ax[2], r["unknown"][k], k)
    viol_axis(ax[2], 60)
    ax[2].set(xlabel=XVIOL, ylabel="Deviation [%]", ylim=(13, 31), title="(c) Unknown patients")
    legend(fig, [ax[2]], 3)
    save(fig, "cinc1")


def chosen(e):
    """(violations, deviation, meets the budget) of the configuration chosen on validation, on the test episodes [%]."""
    v, d, met, _ = episodes(e)
    return 100 * v.mean(), 100 * d.mean(), met


def curve(ax, kind, sb, color, marker, label, ls="-"):
    """Deviation of the chosen configuration for every lambda; hollow if no configuration met the budget."""
    pts = [(lam, *chosen(e)) for lam in LAMS if (e := lookup(kind, lam, sb)) is not None]
    if not pts:
        return
    lam, _, d, met = (np.array(q) for q in zip(*pts))
    ax.plot(lam, d, color=color, ls=ls, marker="none", label="_")
    ax.plot(lam[met], d[met], color=color, marker=marker, ls="none", label="_", markeredgecolor="white", markeredgewidth=0.4)
    ax.plot(lam[~met], d[~met], color=color, marker=marker, ls="none", label="_", markerfacecolor="white")
    ax.plot([], [], color=color, marker=marker, ls=ls, label=label)


def rms_bias(theta, every, beats=400, sigma2=None):
    """Root mean square of the residual bias [ms] of a calibration that ages, over the episode."""
    t = np.arange(beats)
    age = t % every if every else t
    return float(np.sqrt(CAL_ERR**2 + (2.0 * sigma2 * (1.0 - np.exp(-age / theta))).mean()))


def fig2():
    """Where the violations are, what calibration and memory buy for every lambda, and how long a calibration lasts."""
    rows, r, tuned = load("mechanism_ramp.json"), load("monitor/reference_N10_ramp.json"), load("robust/tuned.json") or {}
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.6), gridspec_kw={"width_ratios": [1, 1.25, 1.1]})
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.36, wspace=0.33)
    by_bias(ax[0], rows, 0.9, ("naive", "ibu_d", "ibu_hist", "tdm"))
    ax[0].set(xlabel="Detector bias [SD]", ylabel=XVIOL, title=r"(a) Not calibrated, $\lambda$ = 0.9")
    bl, tr = chosen(r["unknown"]["blind"])[1], chosen(r["unknown"]["true"])[1]
    ax[1].axhline(bl, color=STYLE["blind"][0], ls="--", label="Blind pattern")
    ax[1].axhline(tr, color=INK, label="True count")
    curve(ax[1], "naive", None, STYLE["naive"][0], "v", "Naive")
    for sb in (20.0, 10.0, 0.0):
        curve(ax[1], "naive", sb, AMBER[sb], "v", f"Naive, calibrated ({sb:g} ms)")
    curve(ax[1], "tdm", None, STYLE["tdm"][0], "o", "TDM")
    curve(ax[1], "ibu_hist", None, STYLE["ibu_hist"][0], "P", "IBU recurrent")
    ax[1].set(xlabel=r"Systematic fraction of the error, $\lambda$", ylabel="Deviation [%]", title="(b) Calibrate or remember", ylim=(16.8, 24))
    sig2 = 0.9 * 36.0**2
    if cal := json.loads((OUT / "calibration.json").read_text()):
        sig2 = 0.9 * cal["sigma"] ** 2
    ax[2].axhline(bl, color=STYLE["blind"][0], ls="--")
    ax[2].axhline(tr, color=INK)
    for jump, marker, lab in ((False, "o", "Slow drift"), (True, "X", "Abrupt changes")):
        for rec, face in ((0, None), (100, "white")):
            pts = [(rms_bias(theta, every, sigma2=sig2), chosen(tuned[key(kind, 0.9, None, f"_{name}")])[1]) for name, kind, theta, j, _, every in DRIFTS
                   if kind == "naive" and j == jump and every == rec and key(kind, 0.9, None, f"_{name}") in tuned]
            if pts:
                x, y = zip(*sorted(pts))
                ax[2].plot(x, y, color=AMBER[10.0], marker=marker, ls="none", markerfacecolor=face or AMBER[10.0], label=lab + (", recalibrated" if rec else ""))
    stat = [(sb, chosen(e)[1]) for sb in (0.0, 5.0, 10.0, 20.0) if (e := lookup("naive", 0.9, sb)) is not None]
    ax[2].plot(*zip(*stat), color=AMBER[0.0], marker="v", label="Constant bias")
    ax[2].set(xlabel="RMS residual bias [ms]", ylabel="Deviation [%]", title=r"(c) Ageing calibration, $\lambda$ = 0.9", ylim=(16.8, 24))
    legend(fig, ax, 4)
    save(fig, "cinc2")


def robust_numbers():
    r = load("monitor/reference_N10_ramp.json")["unknown"]
    ref = {k: episodes(r[k])[:2] for k in ("blind", "true")}

    def describe(e, against=()):
        v, d, met, cfg = episodes(e)
        out = {"config": cfg, "met": bool(met), "val_viol": 100 * float(e["val2_viol"][pick(e)[0]]), "viol": 100 * float(v.mean()), "w5": 100 * worst5(v), "dev": 100 * float(d.mean())}
        for name, (v0, d0) in {**ref, **dict(against)}.items():
            out[f"dev_minus_{name}"], out[f"viol_minus_{name}"] = paired(d - d0), paired(v - v0)
        return out

    num: dict = {"sweep": {}, "other": {}, "shift": {}, "mismatch": {}}
    for lam in LAMS:
        tdm = lookup("tdm", lam)
        against = {"tdm": episodes(tdm)[:2]} if tdm else {}
        for kind, sb in (("naive", None), ("naive", 0.0), ("naive", 10.0), ("naive", 20.0), ("tdm", None), ("ibu_hist", None)):
            if (e := lookup(kind, lam, sb)) is not None:
                num["sweep"][key(kind, lam, sb)] = describe(e, against)
    for name, e in (load("robust/tuned.json") or {}).items():
        if name not in num["sweep"]:
            num["other"][name] = describe(e)
    fixed = load("robust/fixed.json") or {}
    for name, e in fixed.items():
        v, d = np.array(e["viol_episodes"]), np.array(e["dev_episodes"])
        row = {"viol": 100 * float(v.mean()), "w5": 100 * worst5(v), "dev": 100 * float(d.mean()), "open": 100 * float(np.mean(e["open_episodes"]))}
        if name.startswith("mismatch"):
            num["mismatch"][name] = row
            continue
        shift = name.split("_")[0]
        if f"{shift}_blind" in fixed:
            b = fixed[f"{shift}_blind"]
            row |= {"dev_minus_blind": paired(d - np.array(b["dev_episodes"])), "viol_minus_blind": paired(v - np.array(b["viol_episodes"]))}
        num["shift"][name] = row
    (OUT / "robust").mkdir(exist_ok=True)
    (OUT / "robust" / "numbers.json").write_text(json.dumps(num, indent=1))
    show(num)


def show(num):
    for part in ("sweep", "other"):
        for k, x in num[part].items():
            print(f"{k:34s} {'' if x['met'] else '(budget not met) '}viol {x['viol']:.2f} w5 {x['w5']:.0f} dev {x['dev']:.2f}  dev-blind {x['dev_minus_blind'][0]:+.2f} "
                  f"[{x['dev_minus_blind'][1]:+.2f}, {x['dev_minus_blind'][2]:+.2f}]" + (f"  dev-tdm {x['dev_minus_tdm'][0]:+.2f} [{x['dev_minus_tdm'][1]:+.2f}, {x['dev_minus_tdm'][2]:+.2f}]" if "dev_minus_tdm" in x else ""))
    for part in ("mismatch", "shift"):
        for k, x in num[part].items():
            print(f"{k:34s} viol {x['viol']:.2f} w5 {x['w5']:.0f} dev {x['dev']:.2f}" + (f"  dev-blind {x['dev_minus_blind'][0]:+.2f}  viol-blind {x['viol_minus_blind'][0]:+.2f}" if "dev_minus_blind" in x else ""))


def main() -> None:
    with plt.rc_context(RC):
        fig1()
        fig2()
    robust_numbers()


if __name__ == "__main__":
    assert set(DONE) <= set(LAMS) and SHIFTS and SHIFT_POLICIES
    main()
