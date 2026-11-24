"""Phase 6.5 analysis (docs/phase6_5_plan.md, section B5): pre-registered comparisons and summary.

    python scripts/analyze_phase6_5.py [--name eval]

Per agent group (seeds) and test set: interquartile mean (IQM) with a 95 % percentile bootstrap CI
over seeds, mean and standard error. Primary hypotheses (Welch, two-sided, Holm over the family):
  P1  TDM vs IBU, % of time in closures > 10 s: hypertension family and random test set
  P2  oracle vs recurrent PPO, same metric, random test set
(P3, the Simplex safety layer, is analysed with its own evaluation.) Probability of improvement
P(X_A < X_B) over all seed pairs. Secondary comparisons are reported without correction.
Writes results/phase6_5/summary.md and analysis.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path("results/phase6_5")
SETS = ("random", "hypertension", "exercise", "hypovolemia")
PRIMARY = "time_closed_gt10s"
METRICS = ("time_closed_gt10s", "time_closed_gt30s", "opening_ratio", "max_closure_s", "violations_per_hour",
           "suction_per_hour", "back_frac", "congestion_frac", "co_abs_err", "speed_changes_per_min", "return_mean")
PRIMARY_TESTS = [("P1-hypertension", "tdm_v2_mlp", "ibu_v2_mlp", "hypertension"),
                 ("P1-random", "tdm_v2_mlp", "ibu_v2_mlp", "random"),
                 ("P2", "oracle", "rnn", "random")]
SECONDARY_TESTS = [("TDM-prop vs IBU", "tdm_det_v2_mlp", "ibu_v2_mlp"), ("TDM vs IBU-hist", "tdm_v2_mlp", "ibu_hist_v2_mlp"),
                   ("IBU-hist vs IBU", "ibu_hist_v2_mlp", "ibu_v2_mlp"), ("TDM vs Naive", "tdm_v2_mlp", "naive_v2_mlp"),
                   ("TDM vs oracle", "tdm_v2_mlp", "oracle"), ("oracle_p vs oracle", "oracle_p", "oracle")]


def group(label: str) -> str:
    return re.sub(r"_s\d+$", "", label)


def iqm(x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def boot_ci(x: np.ndarray, f=iqm, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = [f(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    t = stats.ttest_ind(a, b, equal_var=False)
    p_imp = float(np.mean(a[:, None] < b[None, :]) + 0.5 * np.mean(a[:, None] == b[None, :]))
    return {"diff_mean": float(a.mean() - b.mean()), "p": float(t.pvalue), "prob_improvement": p_imp, "n": [len(a), len(b)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="eval")
    args = ap.parse_args()
    ev = json.loads((OUT / f"{args.name}.json").read_text())
    groups: dict[str, list[str]] = {}
    for label in ev:
        groups.setdefault(group(label), []).append(label)
    def get(d, m):  # horizon-specific keys (violations_per_hour_N3, return_mean_N3, ...)
        return d[m] if m in d else next(v for k, v in d.items() if k.startswith(m + "_N"))

    vals = {g: {s: {m: np.array([get(ev[l][s], m) for l in labels]) for m in METRICS} for s in SETS} for g, labels in groups.items()}

    summary = {}
    for g in groups:
        summary[g] = {s: {m: {"iqm": iqm(v), "ci": boot_ci(v), "mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                              "n": len(v)} for m, v in vals[g][s].items()} for s in SETS}
    tests = []
    for name, a, b, s in PRIMARY_TESTS:
        if a in vals and b in vals:
            tests.append({"test": name, "a": a, "b": b, "set": s, **compare(vals[a][s][PRIMARY], vals[b][s][PRIMARY])})
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
    secondary = []
    for name, a, b in SECONDARY_TESTS:
        if a in vals and b in vals:
            for s in ("random", "hypertension"):
                secondary.append({"test": name, "a": a, "b": b, "set": s, **compare(vals[a][s][PRIMARY], vals[b][s][PRIMARY])})
    (OUT / "analysis.json").write_text(json.dumps({"summary": summary, "primary": tests, "secondary": secondary}, indent=2))

    order = sorted(groups, key=lambda g: (len(groups[g]) == 1, g))
    L = ["# Fase 6.5: resultados (entorno v2)", "",
         "IQM sobre semillas [IC 95 % bootstrap]; los controladores clásicos son deterministas (n = 1). Inicio en la velocidad del clínico.", ""]
    for s in SETS:
        L += [f"## {s}", "", "| Política | n | Tiempo cerrada > 10 s | > 30 s | Razón de apertura | Cierre máx. (s) | Violaciones/h | Succión/h | Reflujo | Congestión | Error de gasto | Retorno |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for g in order:
            r = summary[g][s]

            def cell(m, fmt):
                x = r[m]
                ci = "" if r[m]["n"] < 2 else f" [{fmt.format(x['ci'][0])}, {fmt.format(x['ci'][1])}]"
                return fmt.format(x["iqm"]) + ci

            L.append(f"| {g} | {r[PRIMARY]['n']} | {cell('time_closed_gt10s', '{:.2%}')} | {cell('time_closed_gt30s', '{:.2%}')} | "
                     f"{cell('opening_ratio', '{:.2f}')} | {cell('max_closure_s', '{:.1f}')} | {cell('violations_per_hour', '{:.1f}')} | "
                     f"{cell('suction_per_hour', '{:.2f}')} | {cell('back_frac', '{:.3f}')} | {cell('congestion_frac', '{:.3f}')} | "
                     f"{cell('co_abs_err', '{:.2f}')} | {cell('return_mean', '{:.1f}')} |")
        L.append("")
    L += ["## Hipótesis principales (tiempo con la válvula cerrada > 10 s; Welch bilateral; Holm)", "",
          "| Prueba | A − B | Conjunto | Δ media | p | p (Holm) | P(A < B) | n |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in tests:
        L.append(f"| {t['test']} | {t['a']} − {t['b']} | {t['set']} | {t['diff_mean']:+.3%} | {t['p']:.3g} | {t['p_holm']:.3g} | {t['prob_improvement']:.2f} | {t['n']} |")
    L += ["", "## Comparaciones secundarias (sin corrección)", "", "| Prueba | Conjunto | Δ media | p | P(A < B) | n |", "| --- | --- | --- | --- | --- | --- |"]
    for t in secondary:
        L.append(f"| {t['test']} | {t['set']} | {t['diff_mean']:+.3%} | {t['p']:.3g} | {t['prob_improvement']:.2f} | {t['n']} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
