"""Phase 7 agents: PPO results over training seeds (CPU).

    python scripts/phase7_agents.py [--n 10]
Reads results/phase7/agents.json; writes results/phase7/table_agents_N*.tex and agents_summary_N*.json.

Every agent maximizes the same reward up to the weight r_v of a violation, so agents are compared without choosing anything
on test episodes: cost = deviation + r_v * violations + 0.3 * backflow, the mean per beat without the small speed-change
term. The other number is the smallest deviation among the r_v whose violations are at most 1 % of the beats on average
over seeds; the choice is among three values and it is made on the test episodes, so it favours every agent equally.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from phase7_common import OUT

NAMES = {
    "true": "RM, true labels", "true_rnn": "RM + GRU, true labels", "last@true": "Recurrent, true labels", "window@true": "Window, true labels",
    "none": "Recurrent, no detector", "naive": "RM, Naive", "ibu": "RM, IBU per-beat", "ibu_hist": "RM, IBU recurrent", "tdm": "RM, TDM",
    "filter": "RM, filter", "window": "Window, no RM", "last": "Recurrent, no RM", "press": "Recurrent, pressures",
    "press_mlp": "Memoryless, pressures",
}
BUDGET = 0.01


def rows(n):
    """(lam, kind) -> {rv: list of entries over seeds}; agents with true labels have lam = None."""
    a = json.loads((OUT / "agents.json").read_text())
    g = defaultdict(lambda: defaultdict(list))
    for e in a.values():
        if e["regime"] == "unknown" and e["n"] == n:
            g[(e["lam"], e["kind"])][e["rv"]].append(e)
    return g


def cost(e):
    return e["dev"] + e["rv"] * e["viol"] + 0.3 * e["back"]


def summarize(by_rv):
    out = {}
    for rv, es in sorted(by_rv.items()):
        out[f"{rv:g}"] = {k: [float(np.mean(x)), float(np.min(x)), float(np.max(x))] for k, x in
                          (("viol", [e["viol"] for e in es]), ("dev", [e["dev"] for e in es]), ("worst5", [e["viol_worst5"] for e in es]),
                           ("cost", [cost(e) for e in es]))} | {"seeds": len(es)}
    ok = {rv: x for rv, x in out.items() if x["viol"][0] <= BUDGET}
    best = min(ok, key=lambda rv: ok[rv]["dev"][0]) if ok else None
    return {"rv": out, "budget_rv": best}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    n = p.parse_args().n
    g = rows(n)
    s = {}
    for (lam, kind), by_rv in g.items():
        s.setdefault("none" if lam is None else f"{lam:g}", {})[kind] = summarize(by_rv)
    (OUT / f"agents_summary_N{n}.json").write_text(json.dumps(s, indent=1))
    for lam in sorted(s):
        print(f"== N = {n}, lambda = {lam}")
        for kind, x in s[lam].items():
            cells = "  ".join(f"rv{rv}: {100 * v['viol'][0]:5.2f} {100 * v['dev'][0]:5.1f} w5 {100 * v['worst5'][0]:3.0f} cost {100 * v['cost'][0]:5.1f} "
                              f"[{100 * v['cost'][1]:.1f}, {100 * v['cost'][2]:.1f}] ({v['seeds']})" for rv, v in x["rv"].items())
            b = x["budget_rv"]
            print(f"  {kind:10s} {cells}  budget: {'none' if b is None else f'rv{b}, dev ' + format(100 * x['rv'][b]['dev'][0], '.1f')}")
    lams = [lam for lam in ("0", "0.5", "0.9") if lam in s]
    if not lams:
        return
    lines = [r"\begin{tabular}{l" + "rrrr" * len(lams) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{$\lambda={lam}$}}" for lam in lams) + r" \\",
             "Agent & " + " & ".join(["Viol.", "Worst 5\\%", "Dev.", "Cost"] * len(lams)) + r" \\", r"\midrule"]
    for kind in ("true", "last@true", "window@true", "none", "naive", "ibu", "ibu_hist", "tdm", "filter", "window", "last"):
        cells = []
        for lam in lams:
            src = s.get("none", {}).get(kind.split("@")[0]) if kind in ("true", "none") or "@" in kind else s[lam].get(kind)
            x = src and src["rv"].get("3")
            cells += ["--"] * 4 if not x else [f"{100 * x['viol'][0]:.2f}", f"{100 * x['worst5'][0]:.0f}", f"{100 * x['dev'][0]:.1f}", f"{100 * x['cost'][0]:.1f}"]
        lines.append(NAMES[kind] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"table_agents_N{n}.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
