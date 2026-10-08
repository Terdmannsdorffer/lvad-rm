# lvad-rm

LVAD speed control with noisy reward machines. Course project for IIC3675 (PUC Chile).
Project context, decisions and plan: [HANDOFF.md](HANDOFF.md).

## Model

0D left ventricle + systemic circulation + axial rotary LVAD (Simaan et al. 2009),
reimplemented in JAX:

- volume-form state `[LVV, LAP, AP, AoP, Qt, Qp]`, double-hill elastance, ideal-diode valves;
- pump head with the physically consistent sign (the paper's matrix form with the Table IV
  coefficients is unstable and pumps backwards, see `scripts/validate_simaan.py`);
- fixed-step RK4 inside `lax.scan`, one cardiac beat per call, with per-beat statistics and
  RK4-consistent integrals (`src/lvadrm/integrate.py`).

## Setup (Windows + WSL2, GPU)

```bash
# inside WSL (Ubuntu 22.04)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.venvs/lvad-rm --python 3.12
uv pip install --python ~/.venvs/lvad-rm/bin/python "jax[cuda12]" -e ".[dev]"
```

From Git Bash on Windows, `./run_wsl.sh <command>` runs a command in that venv from the repo root
(for example `./run_wsl.sh python -m pytest -q`).

## Reproduce Phase 1

```bash
python scripts/run_phase1.py      # all validation figures, JSON and results/phase1/summary.md
python -m pytest -q               # model tests (float64, CPU)
```

Validation scripts run in float64 on CPU; `scripts/bench_gpu.py` runs in float32 on the GPU.
