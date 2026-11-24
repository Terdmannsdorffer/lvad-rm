"""Phase 7: what does one calibration of the detector per patient buy? (GPU)

A clinician who measures the opening time with echocardiography in some beats can estimate the detector bias b of the
patient and correct the threshold. What is left is the error of that estimate and the drift of b since then, s_b: the
detector keeps the per-beat noise (1 - lam) sigma^2 and has a residual bias with standard deviation s_b, i.e. it is the
same detector with sigma'^2 = (1 - lam) sigma^2 + s_b^2 and lam' = s_b^2 / sigma'^2. Same controller class, grids,
scenarios, noise seeds, screening and second validation as phase7_monitor.py and phase7_revalidate.py.

    python scripts/phase7_calibrated.py --nominal ramp --lam 0.9 0.5 --sb 0 5 10 20
Writes results/phase7/monitor/calibrated_N10[_ramp].json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
from phase7_common import OUT, detector
from phase7_monitor import (
    BEATS,
    FACTORS,
    N_TEST,
    N_VAL,
    Q_SET,
    TEST_SEED,
    VAL_SEED,
    grid,
    mon,
    run_configs,
    select_and_test,
    threshold_speeds,
)
from phase7_revalidate import N_VAL2
from phase7_revalidate import SEED as VAL2_SEED

from lvadrm.estimators import make_estimator, monitor_fns
from lvadrm.presets import env_v3

N = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, nargs="+", default=[0.9, 0.5])
    ap.add_argument("--sb", type=float, nargs="+", default=[0.0, 5.0, 10.0, 20.0])
    ap.add_argument("--nominal", choices=["flow", "ramp"], default="flow")
    a = ap.parse_args()
    ramp = a.nominal == "ramp"
    cfg = env_v3(N, "unknown", episode_beats=BEATS)
    path = OUT / "monitor" / f"calibrated_N{N}{'_ramp' if ramp else ''}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    configs = grid(FACTORS if ramp else Q_SET, range(1, N), (0.5,))
    kw = dict(ref_val=threshold_speeds(cfg, VAL_SEED, N_VAL), ref_test=threshold_speeds(cfg, TEST_SEED, N_TEST)) if ramp else {}
    ref2 = threshold_speeds(cfg, VAL2_SEED, N_VAL2) if ramp else None
    for lam in a.lam:
        base = detector(lam)
        for sb in a.sb:
            name = f"naive_lam{lam:g}_sb{sb:g}"
            s2 = (1.0 - lam) * base.sigma**2 + sb**2
            det = replace(base, sigma=float(np.sqrt(s2)), lam=float(sb**2 / s2))
            fns = monitor_fns(make_estimator("naive", det, N), cfg, N)
            if name not in res:
                select_and_test(name, fns, mon, configs, cfg, N, res, **kw)
                res[name] |= {"sigma": det.sigma, "lam_eff": det.lam}
                path.write_text(json.dumps(res))
            if "val2_viol" not in res[name]:
                o = run_configs(fns, mon, np.array(res[name]["configs"], float), cfg, N, VAL2_SEED, N_VAL2, ref=ref2)
                res[name] |= {"val2_viol": o["viol"].mean(1).tolist(), "val2_dev": o["dev"].mean(1).tolist()}
                path.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
