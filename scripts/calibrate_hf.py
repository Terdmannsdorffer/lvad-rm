"""Heart-failure patient under LVAD support (HANDOFF section 7), float64 on CPU.

Fits Emax, Emin, V0, Vtot and the pump speed (and optionally Rs) to Fernandez-Zapico et al.
(2024), Table 1. Two variants: Rs fixed to their SVR (0.9) and Rs free (our systemic return
goes to the left atrium, not to a venous node, so our Rs is not exactly their SVR).
The chosen values are copied into src/lvadrm/patients.py.

Writes results/phase2/calibrate_hf.json and results/phase2/hf_patient.png.
"""

from __future__ import annotations

import common

common.setup_cpu_x64()

import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lvadrm.calibration import HF_HR, HF_TARGETS, calibrate_hf_patient, run_beats  # noqa: E402
from lvadrm.integrate import SAMPLE_NAMES, simulate_beats  # noqa: E402
from lvadrm.params import make_params  # noqa: E402

SIM = {"n_seg": 64, "n_inner": 105}
S = {name: i for i, name in enumerate(SAMPLE_NAMES)}


def main() -> None:
    out = {"targets": HF_TARGETS, "fits": {}}
    for fit_rs in (True, False):
        t0 = time.perf_counter()
        r = calibrate_hf_patient(fit_rs=fit_rs, **SIM)
        r["seconds"] = time.perf_counter() - t0
        out["fits"]["rs_free" if fit_rs else "rs_fixed_0.9"] = r
        print("rs_free" if fit_rs else "rs_fixed_0.9", r["params"], r["metrics"], "in range:", sum(r["in_range"].values()), "/ 8")
    chosen = min(out["fits"], key=lambda k: out["fits"][k]["cost"])
    out["chosen"] = chosen
    common.save_json("calibrate_hf.json", out, phase=2)

    # waveforms and PV loop of the chosen patient
    pr = out["fits"][chosen]["params"]
    p = make_params(hr=HF_HR, emax=pr["emax"], emin=pr["emin"], v0=pr["v0"], rs=pr["rs"])
    y, _, _ = run_beats(p, pr["vtot"], pr["omega"], n_beats=30, **SIM)
    n_sub = SIM["n_seg"] * SIM["n_inner"]
    _, _, smp = simulate_beats(y, p, jnp.full(3, pr["omega"]), n_seg=n_sub, n_inner=1)
    smp = np.asarray(smp).reshape(-1, len(SAMPLE_NAMES))
    t = (np.arange(len(smp)) + 1) * 60.0 / HF_HR / n_sub

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(t, smp[:, S["aop"]], label="AoP")
    ax[0].plot(t, smp[:, S["lvp"]], label="LVP")
    ax[0].plot(t, smp[:, S["lap"]], label="LAP")
    ax[0].set(xlabel="time (s)", ylabel="pressure (mmHg)")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, smp[:, S["qp"]], label="pump")
    ax[1].plot(t, smp[:, S["qa"]], label="aortic valve")
    ax[1].set(xlabel="time (s)", ylabel="flow (ml/s)")
    ax[1].legend(fontsize=8)
    beat = slice(0, n_sub)
    ax[2].plot(smp[beat, S["lvv"]], smp[beat, S["lvp"]])
    ax[2].set(xlabel="LVV (ml)", ylabel="LVP (mmHg)", title="PV loop")
    m = out["fits"][chosen]["metrics"]
    fig.suptitle(
        f"HF patient ({chosen}): Emax {pr['emax']:.2f}, Emin {pr['emin']:.3f}, V0 {pr['v0']:.1f}, Vtot {pr['vtot']:.0f}, "
        f"Rs {pr['rs']:.3f}, {pr['omega']:.0f} rpm, HR {HF_HR:.0f} | CO {m['co']:.2f}, QLV {m['qlv']:.2f} l/min, "
        f"AoP {m['aop_max']:.0f}/{m['aop_min']:.0f} ({m['aop_mean']:.0f}), LAP {m['lap_mean']:.1f}, LVV {m['lvv_min']:.0f}-{m['edv']:.0f} ml",
        fontsize=9,
    )
    common.savefig(fig, "hf_patient.png", phase=2)
    plt.close(fig)
    print("chosen:", chosen, "-> wrote results/phase2/calibrate_hf.json")


if __name__ == "__main__":
    main()
