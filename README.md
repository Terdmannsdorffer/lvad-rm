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

## Environment

`lvadrm.env` is a functional, pure-JAX environment (gymnax style): one step is one cardiac beat,
the action is a multiplicative speed change, and the reward comes from a counting reward machine
(`lvadrm.reward_machine`) over the per-beat propositions AVO, SUC, BACK and LOW (`lvadrm.labels`).
Patients and within-episode events come from `lvadrm.scenarios`; the calibrated heart-failure
patient is `lvadrm.patients.HF_PATIENT`. Classical baselines live in `lvadrm.baselines`.

## Reproduce

```bash
python scripts/run_phase1.py      # model validation: results/phase1/summary.md
python scripts/run_phase2.py      # patient, controllers, task difficulty: results/phase2/summary.md
python scripts/run_phase3.py      # PPO agents (oracle, pressure oracle, recurrent, Fernandez-Zapico MDP): results/phase3/summary.md
python scripts/run_phase4.py      # data, AVO detectors, Naive / IBU / TDM inference, belief agents: results/phase4/
python scripts/run_phase5.py      # main experiment, 5 seeds per agent: results/phase5/summary.md
python scripts/collect_data_v2.py && python scripts/run_phase6.py   # robustness, generalization, N ablation: results/phase6/
python scripts/eval_modulation.py && python scripts/run_phase6.py --stages E   # modulation baselines; clinical-level AVO detector
python -m pytest -q               # tests (float64, CPU)
```

Agents (`lvadrm.agents`): PureJaxRL-style PPO and recurrent PPO (GRU) in Flax/Optax, trained on
8192 synchronized environments; `scripts/train_ppo.py --agent {oracle,oracle_p,rnn,mlp,fz,fz4}` trains one.

Validation scripts run in float64 on CPU; benchmarks and environment rollouts run in float32 on the GPU.
Decisions and assumptions: [docs/phase1_decisions.md](docs/phase1_decisions.md), [docs/phase2_decisions.md](docs/phase2_decisions.md), [docs/phase3_decisions.md](docs/phase3_decisions.md), [docs/phase4_decisions.md](docs/phase4_decisions.md), [docs/phase5_decisions.md](docs/phase5_decisions.md), [docs/phase6_decisions.md](docs/phase6_decisions.md).
