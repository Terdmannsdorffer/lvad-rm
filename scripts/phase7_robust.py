"""Phase 7: how far does "calibrate or remember" hold? Same controller, nominal speed from a ramp test, N = 10 (GPU).

Design and predictions: docs/phase7_decisions.md, section 7 (written before the first run). Parts:
    naive     counting rule for every lambda, without calibration and with a residual bias of 0, 10 and 20 ms
    vote      debounced counting rules (an opening counts if at least K of the last W beats are detected open)
    drift     the bias changes during the episode (slowly or abruptly), with and without recalibration
    learned   TDM and recurrent IBU for every lambda with trained models (scripts/phase7_models.py --skip-quality)
    combine   TDM trained for a calibrated detector
    window    the counter predicted from the last 16 beats (no recurrent memory)
    seeds     TDM and recurrent IBU with other training seeds
    mismatch  TDM and recurrent IBU with a detector of another lambda than in training (configurations are not tuned again)
    shift     patients outside the training ranges (configurations are not tuned again)
    shift_combined  the same for TDM trained for a calibrated detector (added after seeing combine and shift)

    python scripts/phase7_robust.py --part naive vote drift
Writes results/phase7/robust/{tuned,fixed}.json. Finished entries are skipped, and the PAUSE file is honoured.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
from phase7_common import OUT, detector, load_models, model_path, pause_point, window_path, write_atomic
from phase7_monitor import (
    BEATS,
    FACTORS,
    N_TEST,
    N_VAL,
    P_SOFT,
    TEST_SEED,
    VAL_SEED,
    bli,
    grid,
    mon,
    run_configs,
    select_and_test,
    threshold_speeds,
)
from phase7_revalidate import N_VAL2
from phase7_revalidate import SEED as VAL2_SEED
from phase7_tables import episodes

from lvadrm.estimators import DriftingDetector, blind_fns, calibrated_detector, make_estimator, monitor_fns
from lvadrm.presets import V3_SCENARIOS, env_v3

N = 10
LAMS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DONE = (0.0, 0.5, 0.9)  # tuned by phase7_monitor.py (and, for 0.5 and 0.9, by phase7_calibrated.py)
SBS = (0.0, 10.0, 20.0)
HARD = grid(FACTORS, range(1, N), (0.5,))
SOFT = grid(FACTORS, (4, 6, 8), P_SOFT)
CFG = env_v3(N, "unknown", episode_beats=BEATS)
DIR = OUT / "robust"
VOTES = ((0.0, None), (0.5, None), (0.9, None), (0.5, 0.0), (0.5, 10.0), (0.9, 10.0))  # (lambda, residual bias)
CAL_ERR = 5.0  # error of one measurement of the bias [ms] (assumption)
# (name, kind, theta [beats], abrupt changes, calibrated at the start, beats between recalibrations), lambda = 0.9
DRIFTS = (
    ("slow10000", "naive", 1e4, False, True, 0),
    ("slow2500", "naive", 2500.0, False, True, 0),
    ("slow800", "naive", 800.0, False, True, 0),
    ("slow250", "naive", 250.0, False, True, 0),
    ("jump2500", "naive", 2500.0, True, True, 0),
    ("jump800", "naive", 800.0, True, True, 0),
    ("slow800_recal100", "naive", 800.0, False, True, 100),
    ("slow250_recal100", "naive", 250.0, False, True, 100),
    ("jump800_recal100", "naive", 800.0, True, True, 100),
    ("slow2500", "tdm", 2500.0, False, False, 0),
    ("slow250", "tdm", 250.0, False, False, 0),
    ("jump800", "tdm", 800.0, True, False, 0),
)
COMBINE = ((0.9, 10.0), (0.9, 20.0), (0.5, 10.0))
SEEDS = ((0.5, 1), (0.9, 1), (0.5, 2), (0.9, 2))
SHIFT_SEED = 31415
SHIFTS = {
    "weaker": dict(emax_scale=(0.55, 0.75)),
    "resistance": dict(rs_scale=(1.3, 1.6)),
    "fast": dict(hr_range=(100.0, 125.0), hr0_range=(100.0, 125.0)),
    "volume": dict(vtot_scale=(0.80, 0.88)),
    "wide": dict(emax_scale=(0.6, 1.4), rs_scale=(0.65, 1.6), hr_range=(55.0, 125.0), hr0_range=(55.0, 125.0), vtot_scale=(0.82, 1.10)),
}
SHIFT_POLICIES = (("blind", None, None), ("true", None, None), ("naive", 0.9, 10.0), ("tdm", 0.9, None), ("ibu_hist", 0.9, None),
                  ("naive", 0.5, 10.0), ("tdm", 0.5, None), ("ibu_hist", 0.5, None))


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def key(kind, lam, sb=None, extra=""):
    return f"{kind}_lam{lam:g}" + ("" if sb is None else f"_sb{sb:g}") + extra


def lookup(kind, lam, sb=None):
    """Entry of a tuned policy, whichever script tuned it (None if it does not exist)."""
    e = read(DIR / "tuned.json").get(key(kind, lam, sb))
    if e is None and sb is None and (OUT / f"monitor/estimators_N{N}_lam{lam:g}_ramp.json").exists():
        e = read(OUT / f"monitor/estimators_N{N}_lam{lam:g}_ramp.json").get(kind)
    if e is None and sb is not None and kind == "naive":
        e = read(OUT / f"monitor/calibrated_N{N}_ramp.json").get(key(kind, lam, sb))
    return e


class Tuner:
    """Screening on 64 episodes, choice on 512 others and report on the 512 test episodes, as in the rest of Phase 7."""

    def __init__(self):
        DIR.mkdir(parents=True, exist_ok=True)
        self.path, self.refs = DIR / "tuned.json", None

    def __call__(self, name, kind, det, models=None):
        res = read(self.path)
        if name in res:
            return
        pause_point()
        if self.refs is None:
            self.refs = [threshold_speeds(CFG, s, m) for s, m in ((VAL_SEED, N_VAL), (TEST_SEED, N_TEST), (VAL2_SEED, N_VAL2))]
        fns = monitor_fns(make_estimator(kind, det, N, models), CFG, N)
        hard = kind == "naive" or kind.startswith("vote")
        new = {}
        select_and_test(name, fns, mon, HARD if hard else SOFT, CFG, N, new, ref_val=self.refs[0], ref_test=self.refs[1])
        o = run_configs(fns, mon, np.array(new[name]["configs"], float), CFG, N, VAL2_SEED, N_VAL2, ref=self.refs[2])
        new[name] |= {"val2_viol": o["viol"].mean(1).tolist(), "val2_dev": o["dev"].mean(1).tolist(), "sigma": det.sigma, "lam_eff": det.lam}
        write_atomic(self.path, json.dumps(read(self.path) | new))


def tuned_parts(parts):
    tune = Tuner()
    for lam in LAMS if "naive" in parts else ():
        if lam not in DONE:
            tune(key("naive", lam), "naive", detector(lam))
        for sb in SBS:
            if sb < np.sqrt(lam) * detector(lam).sigma and lookup("naive", lam, sb) is None:  # larger errors are worse than no measurement
                tune(key("naive", lam, sb), "naive", calibrated_detector(detector(lam), sb))
    for lam, sb in VOTES if "vote" in parts else ():
        det = detector(lam) if sb is None else calibrated_detector(detector(lam), sb)
        for kind in ("vote22", "vote23"):
            tune(key(kind, lam, sb), kind, det)
    for name, kind, theta, jumps, cal, every in DRIFTS if "drift" in parts else ():
        b = detector(0.9)
        det = DriftingDetector(tau=b.tau, sigma=b.sigma, lam=b.lam, p_pos=b.p_pos, p_neg=b.p_neg, theta=theta, jumps=jumps,
                               cal_err=CAL_ERR if cal else None, cal_every=every)
        tune(key(kind, 0.9, None, f"_{name}"), kind, det, load_models(0.9, N) if kind == "tdm" else None)
    for lam in LAMS if "learned" in parts else ():
        if lam not in DONE and model_path(lam, N).exists():
            for kind in ("tdm", "ibu_hist"):
                tune(key(kind, lam), kind, detector(lam), load_models(lam, N))
    for lam, sb in COMBINE if "combine" in parts else ():
        if model_path(lam, N, 2, sb).exists():
            tune(key("tdm", lam, sb), "tdm", calibrated_detector(detector(lam), sb), load_models(lam, N, 2, sb))
    for lam in DONE if "window" in parts else ():
        if window_path(lam, N).exists():
            tune(key("tdm_win", lam), "tdm_win", detector(lam), load_models(lam, N))
    for lam, seed in SEEDS if "seeds" in parts else ():
        if model_path(lam, N, 2, None, seed).exists():
            for kind in ("tdm", "ibu_hist"):
                tune(key(kind, lam, None, f"_seed{seed}"), kind, detector(lam), load_models(lam, N, 2, None, seed))


def fixed(name, fns, prm, config, cfg, seed, res, path):
    """One configuration, already chosen, on 512 episodes of ``cfg``."""
    if name in res:
        return
    pause_point()
    t0 = time.perf_counter()
    o = run_configs(fns, prm, np.array([config], float), cfg, N, seed, N_TEST, ref=threshold_speeds(cfg, seed, N_TEST))
    res[name] = {"config": list(config)} | {f"{k}_episodes": o[k][0].round(5).tolist() for k in ("viol", "dev", "co", "open", "back")}
    v, d = o["viol"][0], o["dev"][0]
    print(f"{name:32s} [{time.perf_counter() - t0:.0f}s] viol {100 * v.mean():.2f}% dev {100 * d.mean():.1f}% w5 {100 * np.sort(v)[-N_TEST // 20 :].mean():.0f}%", flush=True)
    write_atomic(path, json.dumps(res))


def fixed_parts(parts):
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / "fixed.json"
    res = read(path)
    if "mismatch" in parts:
        for train in DONE:
            for test in DONE:
                for kind in ("tdm", "ibu_hist"):
                    if train != test:
                        c = episodes(lookup(kind, train))[3]
                        fns = monitor_fns(make_estimator(kind, detector(test), N, load_models(train, N)), CFG, N)
                        fixed(f"mismatch_{kind}_train{train:g}_test{test:g}", fns, mon, c, CFG, TEST_SEED, res, path)
    policies = (SHIFT_POLICIES if "shift" in parts else ()) + (tuple(("tdm", *c) for c in COMBINE) if "shift_combined" in parts else ())
    if policies:
        ref = read(OUT / f"monitor/reference_N{N}_ramp.json")["unknown"]
        for shift, kw in SHIFTS.items():
            cfg = replace(CFG, scenario=replace(V3_SCENARIOS["unknown"], **kw))
            for kind, lam, sb in policies:
                name = f"{shift}_{kind}" if lam is None else f"{shift}_{key(kind, lam, sb)}"
                c = episodes(ref[kind] if lam is None else lookup(kind, lam, sb))[3]
                if kind == "blind":
                    fixed(name, blind_fns(cfg), bli, c, cfg, SHIFT_SEED, res, path)
                    continue
                det = detector(0.0, perfect=True) if kind == "true" else detector(lam) if sb is None else calibrated_detector(detector(lam), sb)
                models = load_models(lam, N, 2, sb) if kind in ("tdm", "ibu_hist") else None
                fixed(name, monitor_fns(make_estimator(kind, det, N, models), cfg, N), mon, c, cfg, SHIFT_SEED, res, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", required=True,
                    choices=["naive", "vote", "drift", "learned", "combine", "window", "seeds", "mismatch", "shift", "shift_combined"])
    parts = ap.parse_args().part
    tuned_parts(parts)
    fixed_parts(parts)


if __name__ == "__main__":
    main()
