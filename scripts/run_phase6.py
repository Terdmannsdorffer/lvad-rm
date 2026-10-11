"""Phase 6: robustness and generalization (run from the repository root, GPU, ~14 h). Resumable.

    python scripts/run_phase6.py --stages A B C D

A  TDM v2 (Phase 4 data + wide-scenario and on-policy data), TDM agents with it (5 seeds),
   evaluation merged with Phase 5, hypertension diagnosis.
B  generalization of Phase 5/6 agents to wide scenarios and low contractility (no retraining).
C  noise robustness (H2): synthetic detectors iid / Markov with eps 0.1 and 0.3; Naive / IBU / TDM
   agents, 3 seeds.
D  ablation of the RM horizon N in {3, 10} for the oracle and the TDM agent, 3 seeds.
E  clinical-level AVO detector (Clifford et al. 2019: sensitivity 86 %, specificity 93 %), with
   independent or persistent (Markov, rho 0.9) errors: Naive / IBU / TDM agents, 5 seeds, plus the
   detector-in-the-loop modulation rule; references copied from eval.json (same test set).
G  PPO without value clipping (--no-vf-clip) on the 5 runs that collapsed in C and D and on oracle
   seeds 0-2 (collapse fix check).
Run collect_data_v2.py first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

P5 = Path("results/phase5")
P6 = Path("results/phase6")
P5_AGENTS = ["oracle", "oracle_p", "rnn", "naive_mlp", "ibu_mlp", "tdm_mlp"]
NOISE_DETECTORS = ["iid_e0.1", "iid_e0.3", "markov_e0.1_r0.9", "markov_e0.3_r0.9"]
CLINICAL_DETECTORS = ["markov_fn0.14_fp0.07_r0.9", "iid_fn0.14_fp0.07"]
REFERENCES = ["const_pop", "const_clinician", "simaan_minflow", "mod_open_loop", "mod_rule_oracle"]


def py(*cmd):
    print("===", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], check=True)


def train(agent, seed, out_dir, detector=None, tdm_path=None, rm_n=5, tag="", extra=()):
    name = f"{agent}{'_' + detector if detector else ''}{tag}_s{seed}"
    if (Path(out_dir) / name / "params.msgpack").exists():
        return str(Path(out_dir) / name)
    cmd = ["scripts/train_ppo.py", "--agent", agent, "--seed", str(seed), "--out-dir", str(out_dir), "--rm-n", str(rm_n)]
    if detector:
        cmd += ["--detector", detector]
    if tdm_path:
        cmd += ["--tdm-path", tdm_path]
    if tag:
        cmd += ["--tag", tag]
    py(*cmd, *extra)
    return str(Path(out_dir) / name)


def tdm_v2(detector, n=5):
    path = P6 / "tdm" / f"{detector}_N{n}_v2.msgpack"
    if not path.exists():
        py("scripts/train_tdm_v2.py", "--detector", detector, "--n", str(n))
    return str(path)


def p5_dirs(seeds=range(5)):
    return [str(P5 / "runs" / f"{a}_s{s}") for s in seeds for a in P5_AGENTS]


def stage_a():
    path = tdm_v2("mlp", 5)
    v2 = [train("tdm", s, P6 / "runs", "mlp", path, tag="_v2") for s in range(5)]
    if not (P6 / "eval.json").exists():
        py("scripts/eval_agents.py", "--phase", "6", "--name", "eval_v2only", "--no-baselines", "--dirs", *v2)
        merged = json.loads((P5 / "eval.json").read_text())
        new = json.loads((P6 / "eval_v2only.json").read_text())
        for key in ("random_scenarios",):
            merged[key].update(new[key])
        for proto, rows in new["protocols"].items():
            merged["protocols"][proto].update(rows)
        (P6 / "eval.json").write_text(json.dumps(merged, indent=2))
    py("scripts/phase5_aggregate.py", "--phase", "6", "--name", "eval", "--runs", str(P5 / "runs"), str(P6 / "runs"),
       "--title", "Fase 6A: TDM v2 (datos amplios y on-policy)")
    if not (P6 / "hypertension_diagnosis.json").exists():
        py("scripts/diagnose_hypertension.py", "--phase", "6", "--runs", v2[0], str(P5 / "runs" / "tdm_mlp_s0"),
           str(P5 / "runs" / "ibu_mlp_s0"), str(P5 / "runs" / "oracle_s0"))


def stage_b():
    dirs = p5_dirs() + [str(P6 / "runs" / f"tdm_mlp_v2_s{s}") for s in range(5)]
    for scen in ("wide", "lowemax"):
        if not (P6 / f"eval_{scen}.json").exists():
            py("scripts/eval_agents.py", "--phase", "6", "--name", f"eval_{scen}", "--scenarios", scen, "--no-protocols", "--dirs", *dirs)
        py("scripts/phase5_aggregate.py", "--phase", "6", "--name", f"eval_{scen}", "--runs", str(P5 / "runs"), str(P6 / "runs"),
           "--title", f"Fase 6B: generalización ({scen})")


def stage_c():
    out = P6 / "runs_noise"
    dirs = []
    for det in NOISE_DETECTORS:
        path = tdm_v2(det, 5)
        for s in range(3):
            for agent in ("naive", "ibu", "tdm"):
                dirs.append(train(agent, s, out, det, path if agent == "tdm" else None))
    if not (P6 / "eval_noise.json").exists():
        py("scripts/eval_agents.py", "--phase", "6", "--name", "eval_noise", "--no-protocols", "--no-baselines", "--dirs", *dirs)
    py("scripts/phase5_aggregate.py", "--phase", "6", "--name", "eval_noise", "--runs", str(out), "--title", "Fase 6C: robustez al ruido del detector")


def stage_d():
    out = P6 / "runs_N"
    dirs = []
    for n in (3, 10):
        path = tdm_v2("mlp", n)
        for s in range(3):
            dirs.append(train("oracle", s, out, rm_n=n, tag=f"_N{n}"))
            dirs.append(train("tdm", s, out, "mlp", path, rm_n=n, tag=f"_N{n}"))
    dirs += [str(P5 / "runs" / f"oracle_s{s}") for s in range(3)] + [str(P6 / "runs" / f"tdm_mlp_v2_s{s}") for s in range(3)]
    if not (P6 / "eval_N.json").exists():
        py("scripts/eval_agents.py", "--phase", "6", "--name", "eval_N", "--no-protocols", "--no-baselines", "--dirs", *dirs)
    py("scripts/phase5_aggregate.py", "--phase", "6", "--name", "eval_N", "--runs", str(out), str(P5 / "runs"), str(P6 / "runs"),
       "--title", "Fase 6D: ablación del horizonte N")


def stage_e():
    out = P6 / "runs_clinical"
    dirs = []
    for det in CLINICAL_DETECTORS:
        path = tdm_v2(det, 5)
        for s in range(5):
            for agent in ("naive", "ibu", "tdm"):
                dirs.append(train(agent, s, out, det, path if agent == "tdm" else None))
    if not (P6 / "eval_clinical.json").exists():
        py("scripts/eval_agents.py", "--phase", "6", "--name", "eval_clinical", "--no-baselines", "--dirs", *dirs)
        ev = json.loads((P6 / "eval_clinical.json").read_text())
        ref = json.loads((P6 / "eval.json").read_text())
        keep = REFERENCES + [k for k in ref["random_scenarios"] if k.startswith(("oracle_s", "tdm_mlp_v2_s"))]
        for k in keep:
            ev["random_scenarios"][k] = ref["random_scenarios"][k]
            for proto in ev["protocols"]:
                ev["protocols"][proto][k] = ref["protocols"][proto][k]
        (P6 / "eval_clinical.json").write_text(json.dumps(ev, indent=2))
        for det in CLINICAL_DETECTORS:
            py("scripts/eval_modulation.py", "--detector", det, "--families", "rule", "--eval-name", "eval_clinical")
    py("scripts/phase5_aggregate.py", "--phase", "6", "--name", "eval_clinical", "--runs", str(out), str(P5 / "runs"), str(P6 / "runs"),
       "--title", "Fase 6E: detector de nivel clínico (sensibilidad 86 %, especificidad 93 %)")


def stage_g():
    """PPO without value clipping: the 5 runs that collapsed in C and D (all seed 2) and oracle seeds 0-2."""
    out = P6 / "runs_novfclip"
    nv = ("--no-vf-clip",)
    dirs = [train("naive", 2, out, "iid_e0.3", extra=nv),
            train("ibu", 2, out, "iid_e0.3", extra=nv),
            train("ibu", 2, out, "markov_e0.3_r0.9", extra=nv),
            train("oracle", 2, out, rm_n=3, tag="_N3", extra=nv),
            train("tdm", 2, out, "mlp", tdm_v2("mlp", 3), rm_n=3, tag="_N3", extra=nv)]
    dirs += [train("oracle", s, out, extra=nv) for s in range(3)]
    if not (P6 / "eval_novfclip.json").exists():
        py("scripts/eval_agents.py", "--phase", "6", "--name", "eval_novfclip", "--no-baselines", "--dirs", *dirs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["A", "B", "C", "D"])
    args = ap.parse_args()
    for st in args.stages:
        {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d, "E": stage_e, "G": stage_g}[st]()


if __name__ == "__main__":
    main()
