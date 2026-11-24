"""Phase 7 tables: test outcomes of the configuration chosen on validation, with bootstrap intervals (CPU).

    python scripts/phase7_tables.py
Reads results/phase7/monitor and results/phase7/agents.json; writes results/phase7/table_*.tex and tables.json.

Configurations are chosen on validation episodes (scripts/phase7_revalidate.py: 512 episodes; smallest deviation with at
most 1 % of beats in violation; the fewest violations if none complies) and reported on the 512 test episodes. Intervals are percentile bootstraps
over test episodes; differences are paired, because all the policies share patients and noise.
"""

from __future__ import annotations

import json

import numpy as np
from phase7_common import OUT

LAMS = (0.0, 0.5, 0.9)
NAMES = {
    "constant": "Constant speed", "blind": "Blind periodic pattern", "naive": "Naive", "ibu_d": r"IBU, $P(\mathrm{open}\mid d)$",
    "ibu": "IBU, per-beat classifier", "ibu_hist": "IBU, recurrent classifier", "tdm": "TDM", "filter": "Filter (trait, counter)",
    "true": "True counter",
}
BUDGET = 0.01


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def pick(e, budget=BUDGET):
    key = "val2" if "val2_viol" in e else "val"
    v, d = np.array(e[f"{key}_viol"]), np.array(e[f"{key}_dev"])
    ok = v <= budget
    return (int(np.argmin(np.where(ok, d, 9.0))), True) if ok.any() else (int(np.argmin(v)), False)


def worst5(x):
    return float(np.sort(x)[-max(1, len(x) // 20) :].mean())


def boot(x, rng, stat=np.mean, b=2000):
    x = np.asarray(x)
    r = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(b)]
    return [float(stat(x)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def episodes(e, budget=BUDGET):
    i, met = pick(e, budget)
    return np.array(e["viol_episodes"][i]), np.array(e["dev_episodes"][i]), met, e["configs"][i]


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    for sfx, name in (("", "flow"), ("_ramp", "ramp")):
        r = load(f"monitor/reference_N10{sfx}.json")
        if not r or "unknown" not in r:
            continue
        data = {}
        for lam in LAMS:
            es = load(f"monitor/estimators_N10_lam{lam:g}{sfx}.json") or {}
            data[f"{lam:g}"] = {"constant": r["unknown"]["constant"], "blind": r["unknown"]["blind"]} | es | {"true": r["unknown"]["true"]}
        res = out[name] = {}
        for lam, rows in data.items():
            ref = {k: episodes(rows[k]) for k in rows}
            res[lam] = {}
            for k, (v, d, met, cfg) in ref.items():
                res[lam][k] = dict(met=met, config=cfg, viol=boot(v, rng), dev=boot(d, rng), worst5=boot(v, rng, worst5),
                                   viol_minus_true=boot(v - ref["true"][0], rng), dev_minus_true=boot(d - ref["true"][1], rng),
                                   dev_minus_blind=boot(d - ref["blind"][1], rng), viol_minus_blind=boot(v - ref["blind"][0], rng))
        kinds = [k for k in NAMES if any(k in res[lam] for lam in res)]
        lines = [r"\begin{tabular}{l" + "rrr" * len(res) + "}", r"\toprule",
                 " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$\lambda={lam}$}}" for lam in res) + r" \\",
                 "Policy & " + " & ".join(["Viol.", "Worst 5\\%", "Dev."] * len(res)) + r" \\", r"\midrule"]
        for k in kinds:
            cells = []
            for lam in res:
                x = res[lam].get(k)
                cells += ["--"] * 3 if x is None else [f"{100 * x['viol'][0]:.2f}", f"{100 * x['worst5'][0]:.0f}", f"{100 * x['dev'][0]:.1f}"]
            lines.append(NAMES[k] + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT / f"table_{name}.tex").write_text("\n".join(lines) + "\n")
    (OUT / "tables.json").write_text(json.dumps(out, indent=1))
    for name, res in out.items():
        for lam, rows in res.items():
            print(f"== {name} lambda {lam}")
            for k, x in rows.items():
                print(f"  {k:9s} {'' if x['met'] else '(no config met the budget) '}viol {100 * x['viol'][0]:.2f} [{100 * x['viol'][1]:.2f}, {100 * x['viol'][2]:.2f}]  "
                      f"worst5 {100 * x['worst5'][0]:.0f}  dev {100 * x['dev'][0]:.1f} [{100 * x['dev'][1]:.1f}, {100 * x['dev'][2]:.1f}]  "
                      f"dev-blind {100 * x['dev_minus_blind'][0]:+.1f} [{100 * x['dev_minus_blind'][1]:+.1f}, {100 * x['dev_minus_blind'][2]:+.1f}]")


if __name__ == "__main__":
    main()
