"""Calibrated base patients.

HF_PATIENT: heart-failure patient under LVAD support, fitted to Fernandez-Zapico et al. (2024)
Table 1 at HR 80 by scripts/calibrate_hf.py (variant "rs_free"; values copied from
results/phase2/calibrate_hf.json and rounded). The pump speed is the operating point of our
axial pump that reproduces their HVAD-supported resting patient.

SIMAAN_PUMP_HEART: the heart of Simaan et al. (2009) LVAD simulations as calibrated in Phase 1
(HR 60, Emax 2, Emin 0.06, V0 10, Vtot fitted to Fig. 5; see docs/phase1_decisions.md).
"""

from lvadrm.scenarios import Patient

HF_PATIENT = Patient(emax=1.2312, emin=0.12181, v0=73.195, vtot=345.44, rs=0.79298, hr=80.0, omega=8239.7)

SIMAAN_PUMP_HEART = Patient(emax=2.0, emin=0.06, v0=10.0, vtot=287.29, rs=1.0, hr=60.0, omega=13600.0)
