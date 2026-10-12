"""Retrain every PPO agent of Phases 5 and 6 without value clipping (Phase 6G) and repeat the
evaluations (run from the repository root, GPU, ~33 h). Resumable.

    python scripts/run_noclip.py [--stages main F C D B]

main  Phase 5 agents (oracle, oracle_p, rnn, Naive / IBU / TDM with the MLP detector) and TDM v2
      agents (6A), 5 seeds; evaluation with protocols; hypertension diagnosis.
F     clinical-level AVO detector (6F): Naive / IBU / TDM, 2 detectors, 5 seeds.
C     noise sweep (6C): 4 synthetic detectors, Naive / IBU / TDM, 3 seeds.
D     horizon ablation (6D): oracle and TDM with N in {3, 10}, 3 seeds.
B     generalization (6B): main agents on wide and low-contractility scenarios (evaluation only).

Everything goes to results/phase6/noclip/ with the same file names as Phases 5 and 6. Reused as is,
because they do not depend on PPO: detectors, Phase 4 TDM and TDM v2 models. Copied instead of
retrained: the stage G runs (results/phase6/runs_novfclip, same settings). Copied from the clipped
evaluations, because they are deterministic and use the same test sets: classical baselines and
modulation rules.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_phase6 import CLINICAL_DETECTORS, NOISE_DETECTORS, py, tdm_v2, train  # noqa: E402

P6 = Path("results/phase6")
ROOT = P6 / "noclip"
PHASE = "6/noclip"
NV = ("--no-vf-clip",)
SEEDS = range(5)
MAIN = [("oracle", None, None, ""), ("oracle_p", None, None, ""), ("rnn", None, None, ""),
        ("naive", "mlp", None, ""), ("ibu", "mlp", None, ""), ("tdm", "mlp", None, ""), ("tdm", "mlp", "v2", "_v2")]


def copy_stage_g(names: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        if not (out / name).exists():
            shutil.copytree(P6 / "runs_novfclip" / name, out / name)


def merge_references(name: str, source: Path, keys) -> None:
    """Copy entries (random scenarios and protocols) from another evaluation file into ROOT/name.json."""
    ev = json.loads((ROOT / f"{name}.json").read_text())
    ref = json.loads(source.read_text())
    for k in keys:
        ev["random_scenarios"][k] = ref["random_scenarios"][k]
        for proto in ev.get("protocols", {}):
            ev["protocols"][proto][k] = ref["protocols"][proto][k]
    (ROOT / f"{name}.json").write_text(json.dumps(ev, indent=2))


def main_dirs() -> list[str]:
    out = ROOT / "runs"
    dirs = []
    for s in SEEDS:
        for agent, det, v2, tag in MAIN:
            dirs.append(train(agent, s, out, det, tdm_v2("mlp", 5) if v2 else None, tag=tag, extra=NV))
    return dirs


def stage_main() -> None:
    copy_stage_g([f"oracle_s{s}" for s in range(3)], ROOT / "runs")
    dirs = main_dirs()
    if not (ROOT / "eval.json").exists():
        if not (ROOT / "eval_agents_only.json").exists():
            py("scripts/eval_agents.py", "--phase", PHASE, "--name", "eval_agents_only", "--no-baselines", "--dirs", *dirs)
        shutil.copy(ROOT / "eval_agents_only.json", ROOT / "eval.json")
        merge_references("eval", P6 / "eval.json",
                         ["const_pop", "const_clinician", "simaan_minflow", "mod_open_loop", "mod_rule_mlp", "mod_rule_oracle"])
    py("scripts/phase5_aggregate.py", "--phase", PHASE, "--name", "eval", "--runs", str(ROOT / "runs"),
       "--title", "Sin recorte del valor: experimento principal (Fases 5 y 6A)")
    if not (ROOT / "hypertension_diagnosis.json").exists():
        r = ROOT / "runs"
        py("scripts/diagnose_hypertension.py", "--phase", PHASE, "--runs", str(r / "tdm_mlp_v2_s0"), str(r / "tdm_mlp_s0"),
           str(r / "ibu_mlp_s0"), str(r / "oracle_s0"))


def stage_f() -> None:
    out = ROOT / "runs_clinical"
    dirs = []
    for det in CLINICAL_DETECTORS:
        path = tdm_v2(det, 5)
        for s in SEEDS:
            for agent in ("naive", "ibu", "tdm"):
                dirs.append(train(agent, s, out, det, path if agent == "tdm" else None, extra=NV))
    if not (ROOT / "eval_clinical.json").exists():
        py("scripts/eval_agents.py", "--phase", PHASE, "--name", "eval_clinical", "--no-baselines", "--dirs", *dirs)
        main_ev = json.loads((ROOT / "eval.json").read_text())["random_scenarios"]
        agents = [k for k in main_ev if k.startswith(("oracle_s", "tdm_mlp_v2_s"))]
        merge_references("eval_clinical", ROOT / "eval.json",
                         ["const_pop", "const_clinician", "simaan_minflow", "mod_open_loop", "mod_rule_oracle", *agents])
        merge_references("eval_clinical", P6 / "eval_clinical.json", [f"mod_rule_{d}" for d in CLINICAL_DETECTORS])
    py("scripts/phase5_aggregate.py", "--phase", PHASE, "--name", "eval_clinical", "--runs", str(out), str(ROOT / "runs"),
       "--title", "Sin recorte del valor: detector de nivel clínico (sensibilidad 86 %, especificidad 93 %)")


def stage_c() -> None:
    out = ROOT / "runs_noise"
    copy_stage_g(["naive_iid_e0.3_s2", "ibu_iid_e0.3_s2", "ibu_markov_e0.3_r0.9_s2"], out)
    dirs = []
    for det in NOISE_DETECTORS:
        path = tdm_v2(det, 5)
        for s in range(3):
            for agent in ("naive", "ibu", "tdm"):
                dirs.append(train(agent, s, out, det, path if agent == "tdm" else None, extra=NV))
    if not (ROOT / "eval_noise.json").exists():
        py("scripts/eval_agents.py", "--phase", PHASE, "--name", "eval_noise", "--no-protocols", "--no-baselines", "--dirs", *dirs)
    py("scripts/phase5_aggregate.py", "--phase", PHASE, "--name", "eval_noise", "--runs", str(out),
       "--title", "Sin recorte del valor: robustez al ruido del detector")


def stage_d() -> None:
    out = ROOT / "runs_N"
    copy_stage_g(["oracle_N3_s2", "tdm_mlp_N3_s2"], out)
    dirs = []
    for n in (3, 10):
        path = tdm_v2("mlp", n)
        for s in range(3):
            dirs.append(train("oracle", s, out, rm_n=n, tag=f"_N{n}", extra=NV))
            dirs.append(train("tdm", s, out, "mlp", path, rm_n=n, tag=f"_N{n}", extra=NV))
    r = ROOT / "runs"
    dirs += [str(r / f"oracle_s{s}") for s in range(3)] + [str(r / f"tdm_mlp_v2_s{s}") for s in range(3)]
    if not (ROOT / "eval_N.json").exists():
        py("scripts/eval_agents.py", "--phase", PHASE, "--name", "eval_N", "--no-protocols", "--no-baselines", "--dirs", *dirs)
    py("scripts/phase5_aggregate.py", "--phase", PHASE, "--name", "eval_N", "--runs", str(out), str(r),
       "--title", "Sin recorte del valor: ablación del horizonte N")


def stage_b() -> None:
    dirs = main_dirs()
    for scen in ("wide", "lowemax"):
        if not (ROOT / f"eval_{scen}.json").exists():
            py("scripts/eval_agents.py", "--phase", PHASE, "--name", f"eval_{scen}", "--scenarios", scen, "--no-protocols",
               "--no-baselines", "--dirs", *dirs)
            merge_references(f"eval_{scen}", P6 / f"eval_{scen}.json", ["const_pop", "const_clinician", "simaan_minflow"])
        py("scripts/phase5_aggregate.py", "--phase", PHASE, "--name", f"eval_{scen}", "--runs", str(ROOT / "runs"),
           "--title", f"Sin recorte del valor: generalización ({scen})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["main", "F", "C", "D", "B"])
    args = ap.parse_args()
    for st in args.stages:
        {"main": stage_main, "F": stage_f, "C": stage_c, "D": stage_d, "B": stage_b}[st]()


if __name__ == "__main__":
    main()
