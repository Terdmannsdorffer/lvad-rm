"""Phase 7: second validation (512 episodes, its own seed) of the Pareto configurations of every monitor result (GPU).

The first validation (64 episodes) screens ~100-200 configurations; violations are heavy-tailed, so 64 episodes are too
few to tell whether a configuration meets a 1 % budget. This pass re-evaluates only the screened configurations and
stores val2_viol / val2_dev, which scripts/phase7_tables.py uses to choose. The test episodes are not involved.

    python scripts/phase7_revalidate.py
"""

from __future__ import annotations

import json

import numpy as np
from phase7_common import OUT, detector, load_filter, load_models, model_path
from phase7_monitor import BEATS, bli, const_fns, mon, run_configs, threshold_speeds

from lvadrm.estimators import blind_fns, make_estimator, monitor_fns
from lvadrm.presets import env_v3

SEED, N_VAL2 = 778, 512


def main() -> None:
    for path in sorted((OUT / "monitor").glob("*.json")):
        if "coarse" in path.name:
            continue
        res = json.loads(path.read_text())
        ramp = path.stem.endswith("_ramp")
        n = int(path.stem.split("_N")[1].split("_")[0])
        is_ref = path.stem.startswith("reference")
        lam = None if is_ref else float(path.stem.split("_lam")[1].split("_")[0])
        groups = res.items() if is_ref else [("unknown", res)]
        for regime, rows in groups:
            todo = [k for k in rows if "val2_viol" not in rows[k]]
            if not todo:
                continue
            cfg = env_v3(n, regime, episode_beats=BEATS)
            w_avo = threshold_speeds(cfg, SEED, N_VAL2)
            models = {"filter": load_filter()}
            if lam is not None and model_path(lam, n).exists():
                models = load_models(lam, n)
            for kind in todo:
                c = np.array(rows[kind]["configs"], float)
                if kind == "constant":
                    o = run_configs(const_fns(), None, c, cfg, n, SEED, N_VAL2, w_avo)
                elif kind == "blind":
                    o = run_configs(blind_fns(cfg), bli, c, cfg, n, SEED, N_VAL2, ref=w_avo if ramp else None)
                else:
                    det = detector(0.0, perfect=True) if kind == "true" else detector(lam)
                    fns = monitor_fns(make_estimator(kind, det, n, models), cfg, n)
                    o = run_configs(fns, mon, c, cfg, n, SEED, N_VAL2, ref=w_avo if ramp else None)
                rows[kind]["val2_viol"], rows[kind]["val2_dev"] = o["viol"].mean(1).tolist(), o["dev"].mean(1).tolist()
                print(path.stem, regime, kind, np.round(100 * o["viol"].mean(1), 2).tolist(), flush=True)
                # the queue may have added estimators to this file in the meantime: merge into the current version
                cur = json.loads(path.read_text())
                tgt = cur[regime] if is_ref else cur
                tgt[kind] |= {k: rows[kind][k] for k in ("val2_viol", "val2_dev")}
                path.write_text(json.dumps(cur))


if __name__ == "__main__":
    main()
