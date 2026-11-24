"""Phase 7 figures and the numbers quoted in the manuscripts (CPU). Every figure is built only from results/phase7.

    python scripts/phase7_figures.py
Writes results/phase7/figures/*.pdf|png and results/phase7/numbers.json. Figures whose data are missing are skipped.
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from phase7_common import OUT  # noqa: E402

FIG = OUT / "figures"
INK, INK2, GRID = "#1c1b1a", "#5f5d58", "#e4e2dc"
# fixed entity -> (color, marker, line style, label); color follows the entity in every figure
STYLE = {
    "constant": ("#8a877f", "s", ":", "Constant speed"),
    "blind": ("#8a877f", "D", "--", "Blind periodic pattern"),
    "true": (INK, "o", "-", "True counter (oracle)"),
    "naive": ("#eda100", "v", "-", "Naive"),
    "ibu_d": ("#eb6834", "s", "-", "IBU, P(AVO|d)"),
    "ibu": ("#e34948", "^", "-", "IBU, per-beat classifier"),
    "ibu_hist": ("#1baa77", "P", "-", "IBU, recurrent classifier"),
    "tdm": ("#2a78d6", "o", "-", "TDM"),
    "filter": ("#4a3aa7", "X", "-", "Filter (trait, counter)"),
    "window": ("#8a877f", "<", "-", "Window, no RM"),
    "last": ("#5f5d58", ">", "-", "Recurrent, no RM"),
    "none": ("#b9b6ad", "d", "--", "Recurrent, no detector"),
    "true_rnn": (INK, "*", "-", "Recurrent + true counter"),
    "press": ("#0e7490", "p", ":", "Recurrent, pressures"),
    "press_mlp": ("#0e7490", "H", "--", "Memoryless, pressures"),
}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False, "pdf.fonttype": 42, "figure.dpi": 150,
})
XVIOL, YDEV = "Beats in violation [%]", "Deviation from target output [%]"
numbers: dict = {}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote", name)


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def symlog_x(ax):
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.5)
    ax.set_xlim(-0.02, 100)
    ax.set_xticks([0, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0", "0.1", "1", "10", "100"])


def front(ax, e, kind, x="viol", **kw):
    c, m, ls, lab = STYLE[kind]
    v, d = 100 * np.array(e[x]), 100 * np.array(e["dev"])
    o = np.argsort(v)
    gap = np.max(np.diff(v[o])) > 50 if kind == "constant" and len(v) > 1 else False  # all or nothing: nothing in between
    ax.plot(v[o], d[o], color=c, marker=m, ls="none" if gap and len(v) < 4 else ls, label=lab, markeredgecolor="white", markeredgewidth=0.4, **kw)


def fig_map():
    m = load("map.json")
    if m is None:
        return
    b = m["named"]["base"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw={"width_ratios": [1.15, 1.0, 1.15], "wspace": 0.42})
    sp, co, op = np.array(b["speeds"]) / 1000, np.array(b["co"]), np.array(b["open"])
    ax[0].plot(sp[op], co[op], color="#2a78d6", lw=2)
    ax[0].plot(sp[~op], co[~op], color="#8a877f", lw=2, ls="--")
    ax[0].axvline(b["w_avo"] / 1000, color=INK2, lw=0.8)
    ax[0].axhline(5.0, color=INK2, lw=0.8, ls=":")
    ax[0].text(sp[op].max() - 0.15, 3.45, "opens in\nevery beat", ha="right", color=INK, fontsize=7)
    ax[0].text(b["w_avo"] / 1000 + 0.15, 3.45, "never\nopens", color=INK, fontsize=7)
    ax[0].text(6.05, 5.06, "target 5 l/min", fontsize=7, color=INK2)
    ax[0].set(xlabel="Constant pump speed [krpm]", ylabel="Cardiac output [l/min]", xlim=(6, 10.5), ylim=(3.2, 6.3), title="(a) Threshold map, base patient")
    r = b["reference"]["10"]
    seq = np.array(r["seq"])[:30] / 1000
    ax[1].step(range(len(seq)), seq, where="post", color="#2a78d6")
    ax[1].axhline(b["w_avo"] / 1000, color=INK2, lw=0.8)
    ax[1].text(29.5, b["w_avo"] / 1000 - 0.02, r"$\omega_{AVO}$", ha="right", va="top", fontsize=7, color=INK2)
    ax[1].set(xlabel="Beat", ylabel="Speed [krpm]", title="(b) Best pattern, N = 10")
    ns = [3, 5, 10, 20]
    gains = np.array([[p["reference"][str(n)]["gain_pct"] for n in ns] for p in m["named"].values()])
    ax[2].plot(range(4), gains.T, color="#b9b6ad", lw=0.9)
    ax[2].plot(range(4), gains[0], color="#2a78d6", lw=2, marker="o")
    for i, g in enumerate(gains[0]):
        ax[2].annotate(f"+{g:.0f} %", (i, g), textcoords="offset points", xytext=(-4, 5), ha="right", fontsize=7)
    ax[2].set_xticks(range(4), [str(n) for n in ns])
    ax[2].set(xlabel="N (beats allowed without an opening)", ylabel="Output gain over best\nalways-opening speed [%]", xlim=(-0.5, 3.2), title="(c) Counting pays (8 patients)")
    save(fig, "fig1_map")
    numbers["gain_base"] = dict(zip(map(str, ns), gains[0].round(1).tolist()))
    numbers["gain_min"], numbers["gain_max"] = (dict(zip(map(str, ns), f.round(1).tolist())) for f in (gains.min(0), gains.max(0)))
    ra = np.array(m["random"]["co_avo"])
    numbers["frac_need_counting"] = float((ra < 5).mean())
    numbers["w_avo_base"], numbers["co_avo_base"] = b["w_avo"], b["co_avo"]


def at_budget(e, budget=0.01):
    """Smallest deviation among the tested configurations with a mean violation rate within the budget."""
    v, d = np.array(e["viol"]), np.array(e["dev"])
    ok = v <= budget
    return float(d[ok].min()) if ok.any() else None


def worst_at_budget(e, budget):
    v, d = np.array(e["viol"]), np.array(e["dev"])
    ok = v <= budget
    return float(np.array(e["viol_worst5"])[np.argmin(np.where(ok, d, 9))]) if ok.any() else None


def agents_by(regime, lam, n=10):
    a = load("agents.json") or {}
    g = defaultdict(list)
    for e in a.values():
        if e["regime"] == regime and e["lam"] == lam and e["n"] == n:
            g[(e["kind"], e["rv"])].append(e)
    out = defaultdict(list)
    for (kind, rv), es in sorted(g.items()):
        out[kind].append({"rv": rv, "viol": np.array([e["viol"] for e in es]), "dev": np.array([e["dev"] for e in es]),
                          "w5": np.array([e["viol_worst5"] for e in es]), "n": len(es), "back": np.array([e["back"] for e in es])})
    return out


def agent_curve(ax, pts, kind, key="viol"):
    c, m, ls, lab = STYLE[kind]
    x = np.array([100 * p[key].mean() for p in pts])
    y = np.array([100 * p["dev"].mean() for p in pts])
    ax.plot(x, y, color=c, marker=m, ls=ls, label=lab, markeredgecolor="white", markeredgewidth=0.4)
    for p in pts:
        if p["n"] > 1:
            ax.scatter(100 * p[key], 100 * p["dev"], color=c, s=4, alpha=0.45, lw=0)


def fig_ladder():
    r = load("monitor/reference_N10_ramp.json") or load("monitor/reference_N10.json")
    if not r or "unknown" not in r:
        return
    for name, size, leg in (("fig2_ladder", (6.6, 2.5), -0.13), ("fig2_ladder_col", (3.45, 1.9), -0.27)):
        fig, ax = plt.subplots(1, 2, figsize=size, sharey=True, gridspec_kw={"wspace": 0.1})
        for a, regime, t in zip(ax, ("fixed", "unknown"), ("(a) Known patient", "(b) Unknown, changing")):
            for k in ("constant", "blind", "true"):
                front(a, r[regime][k], k)
            symlog_x(a)
            a.set(xlabel=XVIOL, title=t)
            for sfx in ("", "_ramp"):
                q = load(f"monitor/reference_N10{sfx}.json")
                if q and regime in q:
                    numbers[f"ladder_{regime}{sfx}"] = {f"{100 * bud:g}%": {k: at_budget(q[regime][k], bud) for k in ("constant", "blind", "true")} for bud in (0.001, 0.005, 0.01)}
        ax[0].set_ylabel(YDEV)
        h, lab = ax[1].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, leg), columnspacing=0.8, handlelength=1.6)
        save(fig, name)


KINDS = ("naive", "ibu_d", "ibu", "ibu_hist", "tdm", "filter")
LAMS = (0.0, 0.5, 0.9)


def fig_lambda():
    for sfx in ("", "_ramp"):
        r = load(f"monitor/reference_N10{sfx}.json")
        es = {lam: load(f"monitor/estimators_N10_lam{lam:g}{sfx}.json") for lam in LAMS}
        if not r or not all(es.values()):
            continue
        fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.5), sharey=True, gridspec_kw={"wspace": 0.08, "hspace": 0.42})
        for ax, key, xl in zip(axs, ("viol", "viol_worst5"), (XVIOL, "Worst 5 % of patients [%]")):
            for a, lam in zip(ax, LAMS):
                front(a, r["unknown"]["true"], "true", x=key)
                for k in KINDS:
                    if k in es[lam]:
                        front(a, es[lam][k], k, x=key)
                symlog_x(a)
                a.set(xlabel=xl, ylim=(13.5, 23.5))
                if key == "viol":
                    a.set_title(rf"$\lambda$ = {lam:g}" + (" (independent errors)" if lam < 0.1 else " (mostly systematic)" if lam > 0.8 else ""))
            ax[0].set_ylabel(YDEV)
        h, lab = axs[0, -1].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
        save(fig, f"fig3_lambda{sfx}")
        for lam in LAMS:
            e = es[lam] | {"true": r["unknown"]["true"]}
            numbers.setdefault(f"lambda{sfx}", {})[f"{lam:g}"] = {
                k: {"min_viol": float(min(v["viol"]))} | {f"dev@{100 * b:g}%": at_budget(v, b) for b in (0.005, 0.01, 0.02, 0.05)}
                | {f"w5@{100 * b:g}%": worst_at_budget(v, b) for b in (0.01, 0.02, 0.05)} for k, v in e.items()}


def fig_cinc():
    """One-row version with the references, for the four-page paper (ramp-test nominal speed)."""
    r = load("monitor/reference_N10_ramp.json")
    es = {lam: load(f"monitor/estimators_N10_lam{lam:g}_ramp.json") for lam in LAMS}
    if not r or not all(es.values()):
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.15), sharey=True, gridspec_kw={"wspace": 0.08})
    for a, lam, t in zip(axs, LAMS, ("independent errors", "half systematic", "mostly systematic")):
        for k in ("blind", "true"):
            front(a, r["unknown"][k], k)
        for k in ("naive", "ibu_d", "ibu_hist", "tdm"):
            front(a, es[lam][k], k)
        symlog_x(a)
        a.set(xlabel=XVIOL, xlim=(-0.02, 30), ylim=(16, 25.5), title=rf"$\lambda$ = {lam:g} ({t})")
        a.set_xticks([0, 0.1, 1, 10])
        a.set_xticklabels(["0", "0.1", "1", "10"])
    axs[0].set_ylabel("Deviation from\ntarget output [%]")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.22), columnspacing=1.2, handlelength=1.8)
    save(fig, "figB_lambda")


def fig_agents():
    if not load("agents.json"):
        return
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.5), sharey=True, gridspec_kw={"wspace": 0.08, "hspace": 0.42})
    drew = False
    tr = agents_by("unknown", None)
    for ax, key, xl in zip(axs, ("viol", "w5"), (XVIOL, "Worst 5 % of patients [%]")):
        for a, lam in zip(ax, LAMS):
            ag = agents_by("unknown", lam)
            for k in ("true", "none"):
                if k in tr:
                    agent_curve(a, tr[k], k, key)
            for k in ("naive", "ibu", "tdm", "filter", "window", "last"):
                if k in ag:
                    agent_curve(a, ag[k], k, key)
                    drew = True
            symlog_x(a)
            a.set_xlabel(xl)
            if key == "viol":
                a.set_title(rf"$\lambda$ = {lam:g}" + (" (independent errors)" if lam < 0.1 else " (mostly systematic)" if lam > 0.8 else ""))
        ax[0].set_ylabel(YDEV)
    h = {lab: q for a in axs.flat for q, lab in zip(*a.get_legend_handles_labels())}
    fig.legend(h.values(), h.keys(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    if drew:
        save(fig, "fig4_agents")
    plt.close(fig)


def fig_theory():
    t = load("theory.json")
    if t is None:
        return
    s, lam = np.array(t["s"]), np.array(t["lam"])
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 2.1), gridspec_kw={"width_ratios": [1, 1, 0.05, 0.28, 1.1], "wspace": 0.22})
    axs[3].axis("off")
    ax = [axs[0], axs[1], axs[4]]
    for a, k in zip(ax[:2], (5, 10)):
        z = np.array(t[f"log_ratio_k{k}"]) / np.log(10)
        im = a.pcolormesh(s, lam, z, cmap="RdBu", norm=TwoSlopeNorm(0, vmin=-1.0, vmax=1.0), shading="nearest", rasterized=True)
        bd = t["boundary"][str(k)]
        pts = [(b, l) for b, l in zip(bd, lam[1:]) if b is not None]
        a.plot(*zip(*pts), color=INK, lw=1.4)
        a.grid(False)
        a.set(xlabel="Label persistence s", title=f"({'ab'[k == 10]}) Runs of k = {k}")
        a.text(0.80, 0.12, "underestimates\n(unsafe)", fontsize=7, ha="center")
        a.text(0.51, 0.87, "over", fontsize=7, ha="left", color="white" if k == 10 else INK)
    ax[1].set_yticklabels([])
    ax[0].set_ylabel(r"Error persistence $\lambda$")
    cb = fig.colorbar(im, cax=axs[2], ticks=[-1, 0, 1])
    cb.set_ticklabels(["1/10", "1", "10"])
    axs[2].set_title("IBU/truth", fontsize=7)
    rows = [h for h in t["history"] if h["k"] == 8 and h["s"] == 0.9]
    x = np.arange(len(rows))
    for j, (key, c, lab) in enumerate((("truth", INK, "Truth"), ("ibu", "#e34948", "IBU, per-step"), ("ibuh", "#1baa77", "IBU, exact recurrent"))):
        ax[2].bar(x + (j - 1) * 0.27, [h[key] for h in rows], 0.25, color=c, label=lab)
    ax[2].set_xticks(x, [f"{h['lam']:g}" for h in rows])
    ax[2].set(xlabel=r"Error persistence $\lambda$", ylabel="P(8 closed beats)", title="(c) s = 0.9, k = 8")
    ax[2].legend(loc="upper center", bbox_to_anchor=(0.45, -0.27), ncol=3, columnspacing=0.8, handlelength=1.0, fontsize=6)
    ax[2].grid(axis="x", visible=False)
    save(fig, "figA_theory")
    numbers["identity_max_error"] = t["identity_max_error"]


def fig_beliefs():
    b = {lam: load(f"models/beliefs_lam{lam:g}_N10.json") for lam in LAMS}
    if not all(b.values()):
        return
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.1), gridspec_kw={"width_ratios": [1, 1, 1, 1.25], "wspace": 0.35})
    for a, lam in zip(ax[:3], LAMS):
        a.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
        for k in KINDS:
            c, m, ls, lab = STYLE[k]
            rel = np.array(b[lam][k]["reliability"])
            rel = rel[rel[:, 2] >= 100]
            a.plot(rel[:, 0], rel[:, 1], color=c, marker=m, ls=ls, label=lab, ms=3.5, lw=1.1)
        a.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted", title=rf"$\lambda$ = {lam:g}", aspect="equal")
    ax[0].set_ylabel("Observed P(counter = N-1)")
    x = np.arange(3)
    for j, k in enumerate(KINDS):
        c = STYLE[k][0]
        ax[3].bar(x + (j - 2.5) * 0.14, [b[lam][k]["risk_pred"] / b[lam][k]["risk_true"] for lam in LAMS], 0.125, color=c)
    ax[3].axhline(1, color=INK, lw=0.9)
    ax[3].set_xticks(x, [f"{lam:g}" for lam in LAMS])
    ax[3].set(xlabel=r"$\lambda$", title="Mean predicted / true risk")
    ax[3].grid(axis="x", visible=False)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.27))
    save(fig, "figA_beliefs")
    numbers["beliefs"] = {f"{lam:g}": {k: {q: v[q] for q in ("ll", "risk_pred", "risk_true", "ece", "missed", "missed_worst5")} for k, v in b[lam].items()} for lam in LAMS}


def fig_bias(lam="0.5"):
    b = load("bias.json")
    if b is None:
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True, gridspec_kw={"wspace": 0.08})
    titles = (("blind", "(a) Blind pattern"), ("tdm", "(b) Controller driven by TDM"), ("ibu_d", "(c) Controller driven by IBU"))
    for a, (pol, t) in zip(axs, titles):
        row = b[f"{pol}_lam{lam}"]
        k = np.arange(1, len(row["truth"]) + 1)
        a.axhline(1, color=INK, lw=0.9)
        for e in ("naive", "ibu_d", "ibu", "ibu_hist", "tdm"):
            c, m, ls, lab = STYLE[e]
            a.plot(k, np.array(row[e]) / np.array(row["truth"]), color=c, marker=m, ls=ls, label=lab, markeredgecolor="white", markeredgewidth=0.4)
        a.set_yscale("log")
        a.set(xlabel="Length of the run k", title=t, ylim=(0.02, 6))
        a.set_yticks([0.1, 1], ["0.1", "1"])
    axs[0].set_ylabel("Mean belief / truth\nof counter $\\geq$ k")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.17), columnspacing=1.0, handlelength=1.6)
    save(fig, "figA_bias")
    numbers["bias"] = {p: {"truth9": b[f"{p}_lam{lam_}"]["truth"][-1], **{e: b[f"{p}_lam{lam_}"][e][-1] / b[f"{p}_lam{lam_}"]["truth"][-1] for e in KINDS}}
                       for lam_ in ("0", "0.5", "0.9") for p in ("blind", "tdm", "ibu_d")}


BINS = (-9.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 9.0)


def by_bias(ax, rows, lam, kinds):
    """Mean violations per bin of the detector bias of the patient."""
    for k in kinds:
        e = rows.get(f"{k}_lam{lam:g}")
        if e is None:
            continue
        z, v = np.array(e["b"]), 100 * np.array(e["viol"])
        idx = [(z >= lo) & (z < hi) for lo, hi in zip(BINS[:-1], BINS[1:])]
        c, m, ls, lab = STYLE[k]
        ax.plot([z[i].mean() for i in idx], [v[i].mean() for i in idx], color=c, marker=m, ls=ls, label=lab, markeredgecolor="white", markeredgewidth=0.4)
    ax.axvline(0, color=INK2, lw=0.6)


def mechanism_numbers(rows):
    out = {}
    for name, e in rows.items():
        z, v = np.array(e["b"]), np.array(e["viol"])
        out[name] = {"viol": 100 * v.mean(), "below0": 100 * v[z < 0].mean(), "0to1": 100 * v[(z >= 0) & (z < 1)].mean(), "1to1.5": 100 * v[(z >= 1) & (z < 1.5)].mean(),
                     "above1.5": 100 * v[z >= 1.5].mean(), "share_above1": 100 * v[z >= 1].sum() / v.sum(), "patients_above1": 100 * (z >= 1).mean()}
    return out


def fig_mechanism():
    rows = load("mechanism.json")
    if not rows:
        return
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for a, lam, t in zip(ax, (0.5, 0.9), (r"(a) $\lambda$ = 0.5", r"(b) $\lambda$ = 0.9")):
        by_bias(a, rows, lam, ("naive", "ibu_d", "ibu_hist", "tdm", "filter"))
        a.set(xlabel="Bias of the patient's detector [SD]", title=t)
    ax[0].set_ylabel(XVIOL)
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.24), columnspacing=1.0, handlelength=1.6)
    save(fig, "figA_mechanism")
    numbers["mechanism"] = mechanism_numbers(rows)


def fig_calibration():
    rows, cal = load("mechanism_ramp.json"), load("monitor/calibrated_N10_ramp.json")
    r, es = load("monitor/reference_N10_ramp.json"), {lam: load(f"monitor/estimators_N10_lam{lam:g}_ramp.json") for lam in (0.5, 0.9)}
    if not rows or not cal or not r or not all(es.values()):
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw={"wspace": 0.32})
    by_bias(ax[0], rows, 0.9, ("naive", "ibu_d", "ibu_hist", "tdm"))
    ax[0].set(xlabel="Bias of the patient's detector [SD]", ylabel=XVIOL, title=r"(a) No calibration, $\lambda$ = 0.9")
    amber = {20.0: "#d8b46a", 10.0: "#a77a1c", 0.0: "#5c4008"}  # darker = better calibrated
    for a, lam, t in zip(ax[1:], (0.9, 0.5), ("b", "c")):
        front(a, r["unknown"]["blind"], "blind")
        front(a, r["unknown"]["true"], "true")
        for k in ("naive", "tdm"):
            front(a, es[lam][k], k)
        for sb, c in amber.items():
            q = cal.get(f"naive_lam{lam:g}_sb{sb:g}")
            if q:
                v, d = 100 * np.array(q["viol"]), 100 * np.array(q["dev"])
                o = np.argsort(v)
                a.plot(v[o], d[o], color=c, marker="v", ls="-", label=f"Naive, calibrated to {sb:g} ms", markeredgecolor="white", markeredgewidth=0.4)
        symlog_x(a)
        a.set(xlabel=XVIOL, title=rf"({t}) $\lambda$ = {lam:g}", ylim=(16.5, 26.0), xlim=(-0.02, 30))
    ax[1].set_ylabel(YDEV)
    h = {lb: q for a in ax for q, lb in zip(*a.get_legend_handles_labels())}
    fig.legend(h.values(), h.keys(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.27), columnspacing=0.9, handlelength=1.5, fontsize=6.5)
    save(fig, "figB_calibration")
    numbers["mechanism_ramp"] = mechanism_numbers(rows)
    numbers["calibrated_ramp"] = {}
    ref_v, ref_d = episodes_of(r["unknown"]["true"])
    bl_v, bl_d = episodes_of(r["unknown"]["blind"])
    for name, q in cal.items():
        v, d = episodes_of(q)
        numbers["calibrated_ramp"][name] = {"viol": 100 * v.mean(), "dev": 100 * d.mean(), "w5": 100 * np.sort(v)[-len(v) // 20 :].mean(), "met": bool(min(q["val2_viol"]) <= 0.01),
                                            "dev_minus_true": paired(d - ref_d), "viol_minus_true": paired(v - ref_v),
                                            "dev_minus_blind": paired(d - bl_d), "viol_minus_blind": paired(v - bl_v)}


def episodes_of(e):
    from phase7_tables import episodes

    v, d, _, _ = episodes(e)
    return v, d


def paired(x, b=2000):
    rng = np.random.default_rng(0)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(b)]
    return [100 * float(x.mean()), 100 * float(np.percentile(m, 2.5)), 100 * float(np.percentile(m, 97.5))]


def fig_real():
    af, pvc = load("realdata_af.json"), load("realdata_pvc.json")
    if not af or not pvc:
        return
    kinds = ("naive", "ibu", "ibu_w16", "ibu_hist", "tdm")
    lab = {"ibu": "IBU, one interval", "ibu_w16": "IBU, 16-beat window"}
    sty = STYLE | {"ibu_w16": ("#a61e1d", "v", "--", "")}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw={"wspace": 0.28})
    for a, (res, part, t) in zip(ax, ((af, "test", "(a) AF, same database"), (af, "external", "(b) AF, another database"), (pvc, "test", "(c) Ventricular beats"))):
        a.axhline(1, color=INK, lw=0.9)
        n = len(next(iter(res.values()))[part]["truth"])
        ks = np.arange(1, n + 1) if n <= 10 else np.array([1, 2, 3, 5, 10, 15, 20, 25, 30])
        for e in kinds:
            y = np.array([r[part]["est"][e]["ratio"] for r in res.values()])[:, ks - 1]
            c, m, ls, lb = sty[e]
            a.plot(ks, y.mean(0), color=c, marker=m, ls=ls, label=lab.get(e, lb), markeredgecolor="white", markeredgewidth=0.4)
            a.fill_between(ks, y.min(0), y.max(0), color=c, alpha=0.15, lw=0)
        a.set_yscale("log")
        a.set(xlabel="Length of the run k", title=t, ylim=(0.005, 20) if part == "test" and n <= 10 else (0.02, 2))
    ax[0].set_ylabel("Mean belief / truth\nof a run of k beats")
    h, lb = ax[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.18), columnspacing=1.0, handlelength=1.6)
    save(fig, "figA_real")
    num = {}
    for task, res in (("af", af), ("pvc", pvc)):
        for part in ("test", "external"):
            rs = [r[part] for r in res.values() if part in r]
            if not rs:
                continue
            num[f"{task}_{part}"] = {"splits": len(rs), "records": rs[0]["records"], "truth_last": [float(np.mean([r["truth"][-1] for r in rs]))]}
            for e in rs[0]["est"]:
                q = np.array([r["est"][e]["ratio"] for r in rs])
                num[f"{task}_{part}"][e] = {f"ratio_k{k}": [float(q[:, k - 1].mean()), float(q[:, k - 1].min()), float(q[:, k - 1].max())] for k in (1, 2, 3, 5, 10, 30) if k <= q.shape[1]} | {
                    f"negative_ci_k{k}": int(sum(r["est"][e]["beta_hi"][k - 1] < 0 for r in rs)) for k in (3, 5, 30) if k <= q.shape[1]} | {
                    f"positive_ci_k{k}": int(sum(r["est"][e]["beta_lo"][k - 1] > 0 for r in rs)) for k in (3, 5, 30) if k <= q.shape[1]} | {
                    "missed": [float(np.mean([r["est"][e]["missed"] for r in rs])), min(r["est"][e]["missed"] for r in rs), max(r["est"][e]["missed"] for r in rs)],
                    "realized": [float(np.mean([r["est"][e]["realized"] for r in rs])), min(r["est"][e]["realized"] for r in rs), max(r["est"][e]["realized"] for r in rs)]}
            num[f"{task}_{part}"]["persistence"] = {p: {f: [float(np.mean([r["persistence"][p][f] for r in rs])), min(r["persistence"][p][f] for r in rs), max(r["persistence"][p][f] for r in rs)]
                                                        for f in ("rate", "latent")} for p in rs[0]["persistence"]}
        num[f"{task}_lag1"] = [[r["lag1"]["cov_m"], r["lag1"]["cov_C"]] for r in res.values()]
    numbers["real"] = num


def fig_n():
    res = {}
    for n in (5, 10, 20):
        r, e = load(f"monitor/reference_N{n}.json"), load(f"monitor/estimators_N{n}_lam0.5.json")
        if r and e and "unknown" in r:
            res[str(n)] = {k: at_budget(r["unknown"][k]) for k in ("constant", "blind", "true")} | {k: at_budget(v) for k, v in e.items()}
    if res:
        numbers["by_N_budget1"] = res


def main() -> None:
    for f in (fig_map, fig_ladder, fig_lambda, fig_cinc, fig_agents, fig_theory, fig_beliefs, fig_bias, fig_mechanism, fig_calibration, fig_real, fig_n):
        f()
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
