"""Aggregate the Phase 5 evaluation over seeds: mean, standard error, bootstrap 95 % CI, and
Welch t-tests for the comparisons behind H1-H3. Also learning curves (mean +- SE over seeds).

Reads results/phase5/eval.json and the runs' train logs; writes results/phase5/{aggregate.json,
summary.md, agents_bars.png, learning_curves_seeds.png}. With --phase / --name / --runs it aggregates
any evaluation file (e.g. Phase 6) into aggregate_<name>.json and summary_<name>.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

RUNS_DIRS = [Path("results/phase5/runs")]
METRICS = [
    "N5_violations_per_100_beats", "N5_episodes_with_violation", "N3_violations_per_100_beats",
    "N10_violations_per_100_beats", "avo_frac", "back_frac", "suc_frac", "low_frac", "co_mean", "return_mean_N5",
]
COMPARISONS = [  # (a, b, hypothesis)
    ("oracle", "rnn", "H1: RM structure vs memory only"),
    ("tdm_mlp", "ibu_mlp", "H2: TDM vs IBU"),
    ("tdm_mlp", "naive_mlp", "H2: TDM vs Naive"),
    ("tdm_mlp", "rnn", "TDM vs memory only"),
    ("tdm_mlp", "oracle", "H3: pump-only TDM vs RM oracle"),
    ("tdm_mlp", "oracle_p", "H3: pump-only TDM vs pressure oracle"),
    ("oracle_p", "oracle", "pressure vs pump sensors (oracles)"),
]
ORDER = ["const_pop", "const_clinician", "simaan_minflow", "oracle_p", "oracle", "tdm_mlp_v2", "tdm_mlp", "ibu_mlp", "naive_mlp", "rnn"]


def group(label: str) -> str:
    return re.sub(r"_s\d+$", "", label)


def summarize(values: np.ndarray, rng) -> dict:
    v = np.asarray(values, float)
    boot = rng.choice(v, size=(5000, len(v)), replace=True).mean(axis=1) if len(v) > 1 else v
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "n": len(v), "values": v.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=5)
    ap.add_argument("--name", default="eval", help="evaluation file stem")
    ap.add_argument("--runs", nargs="*", default=None, help="directories with the runs' train logs")
    ap.add_argument("--title", default="Fase 5: experimento principal")
    args = ap.parse_args()
    runs_dirs = [Path(r) for r in args.runs] if args.runs else RUNS_DIRS
    out_sfx = "" if args.name == "eval" else "_" + args.name
    ev = common.load_json(f"{args.name}.json", phase=args.phase)
    rng = np.random.default_rng(0)
    groups: dict[str, list[str]] = {}
    for label, m in ev["random_scenarios"].items():
        groups.setdefault(group(label), []).append(label)
        # the RM return is stored as return_mean_N<n> with the agent's own N; report it under one key
        if "return_mean_N5" not in m:
            m["return_mean_N5"] = next(v for k, v in m.items() if k.startswith("return_mean_N"))
    agg = {"random_scenarios": {}, "protocols": {}, "comparisons": []}
    for g, labels in groups.items():
        agg["random_scenarios"][g] = {m: summarize([ev["random_scenarios"][lb][m] for lb in labels], rng) for m in METRICS}
        for proto, rows in ev.get("protocols", {}).items():
            agg["protocols"].setdefault(proto, {})[g] = summarize([rows[lb]["N5_violations_per_100_beats"] for lb in labels], rng)
    comparisons = list(COMPARISONS)
    for g in agg["random_scenarios"]:  # TDM vs IBU / Naive for every detector (e.g. the noise sweep)
        if g.startswith("tdm_") and g != "tdm_mlp":
            det = g.removeprefix("tdm_")
            comparisons += [(g, f"{other}_{det}", f"H2 ({det}): TDM vs {other}") for other in ("ibu", "naive")]
    comparisons += [("tdm_mlp_v2", "tdm_mlp", "TDM v2 vs Phase 4 TDM"), ("tdm_mlp_v2", "ibu_mlp", "H2: TDM v2 vs IBU"),
                    ("tdm_mlp_v2", "oracle", "H3: TDM v2 vs RM oracle")]
    for a, b, hyp in comparisons:
        if a not in agg["random_scenarios"] or b not in agg["random_scenarios"]:
            continue
        row = {"a": a, "b": b, "hypothesis": hyp}
        for m in ("N5_violations_per_100_beats", "co_mean", "avo_frac"):
            va, vb = agg["random_scenarios"][a][m]["values"], agg["random_scenarios"][b][m]["values"]
            t = stats.ttest_ind(va, vb, equal_var=False)
            row[m] = {"diff": float(np.mean(va) - np.mean(vb)), "p_welch": float(t.pvalue)}
        agg["comparisons"].append(row)
    common.save_json(f"aggregate{out_sfx}.json", agg, phase=args.phase)

    order = [g for g in ORDER if g in agg["random_scenarios"]] + [g for g in agg["random_scenarios"] if g not in ORDER]
    R = agg["random_scenarios"]

    def ms(g, m, fmt="{:.2f}", scale=1.0):
        s = R[g][m]
        return (fmt + " ± " + fmt).format(scale * s["mean"], scale * s["se"])

    L = [f"# {args.title} (media ± error estándar sobre semillas)", "",
         f"{ev['n_test']} escenarios de prueba no vistos ({ev.get('scenarios', 'default')}), 300 latidos, r_back = -0.1. Política greedy; "
         "inicio en la velocidad del clínico (const_pop: velocidad poblacional). Las líneas base son deterministas (n = 1).", "",
         "| Política | n | Violaciones / 100 latidos (N=5) | Episodios con violación | N=3 | N=10 | AVO | BACK | LOW | Gasto | Retorno RM |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in order:
        L.append(f"| {g} | {R[g]['co_mean']['n']} | {ms(g, 'N5_violations_per_100_beats')} | {ms(g, 'N5_episodes_with_violation', '{:.0f}', 100)}% | "
                 f"{ms(g, 'N3_violations_per_100_beats')} | {ms(g, 'N10_violations_per_100_beats')} | {ms(g, 'avo_frac', '{:.0f}', 100)}% | "
                 f"{ms(g, 'back_frac', '{:.1f}', 100)}% | {ms(g, 'low_frac', '{:.0f}', 100)}% | {ms(g, 'co_mean')} | {ms(g, 'return_mean_N5', '{:.1f}')} |")
    if agg["protocols"]:
        L += ["", "## Protocolos (1206 latidos): violaciones / 100 latidos (N = 5)", "",
              "| Política | " + " | ".join(agg["protocols"]) + " |", "| --- |" + " --- |" * len(agg["protocols"])]
        for g in order:
            L.append(f"| {g} | " + " | ".join(f"{agg['protocols'][p][g]['mean']:.2f} ± {agg['protocols'][p][g]['se']:.2f}" for p in agg["protocols"]) + " |")
    L += ["", "## Comparaciones (t de Welch sobre semillas)", "",
          "| Hipótesis | A − B | Δ violaciones / 100 (p) | Δ gasto (p) | Δ AVO (p) |", "| --- | --- | --- | --- | --- |"]
    for c in agg["comparisons"]:
        cells = [f"{c[m]['diff']:+.3f} ({c[m]['p_welch']:.3g})" for m in ("N5_violations_per_100_beats", "co_mean", "avo_frac")]
        L.append(f"| {c['hypothesis']} | {c['a']} − {c['b']} | " + " | ".join(cells) + " |")
    (common.results_dir(args.phase) / f"summary{out_sfx}.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # bars
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a_, (m, title) in zip(ax, (("N5_violations_per_100_beats", "violations / 100 beats (N = 5)"), ("co_mean", "mean cardiac output (l/min)"),
                                   ("avo_frac", "fraction of beats with AVO"))):
        a_.bar(range(len(order)), [R[g][m]["mean"] for g in order], yerr=[R[g][m]["se"] for g in order], capsize=3)
        a_.set_xticks(range(len(order)))
        a_.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        a_.set_title(title)
    ax[0].set_yscale("log")
    common.savefig(fig, f"agents_bars{out_sfx}.png", phase=args.phase)
    plt.close(fig)

    # learning curves over seeds
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for g in order:
        logs = [json.loads((rd / lb / "train_log.json").read_text()) for lb in groups[g] for rd in runs_dirs if (rd / lb / "train_log.json").exists()]
        if not logs:
            continue
        beats = np.array([x["beats"] for x in logs[0]]) / 1e6
        for a_, key in zip(ax, ("r_env", "violations_per_100")):
            ys = np.array([np.convolve([x[key] for x in lg], np.ones(12) / 12, mode="valid") for lg in logs])
            mu, se = ys.mean(0), ys.std(0, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else 0 * ys.mean(0)
            a_.plot(beats[11:], mu, label=g)
            a_.fill_between(beats[11:], mu - se, mu + se, alpha=0.2)
    ax[0].set(xlabel="training beats (M)", title="RM reward per beat (mean ± SE over seeds)")
    ax[1].set(xlabel="training beats (M)", title="violations per 100 beats (training, stochastic policy)", yscale="log")
    ax[0].legend(fontsize=7)
    common.savefig(fig, f"learning_curves_seeds{out_sfx}.png", phase=args.phase)
    plt.close(fig)
    print(f"wrote results/phase{args.phase}/aggregate{out_sfx}.json and summary{out_sfx}.md")


if __name__ == "__main__":
    main()
