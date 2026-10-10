# Fase 6B: generalización (lowemax) (media ± error estándar sobre semillas)

2048 escenarios de prueba no vistos (lowemax), 300 latidos, r_back = -0.1. Política greedy; inicio en la velocidad del clínico (const_pop: velocidad poblacional). Las líneas base son deterministas (n = 1).

| Política | n | Violaciones / 100 latidos (N=5) | Episodios con violación | N=3 | N=10 | AVO | BACK | LOW | Gasto | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const_pop | 1 | 20.00 ± 0.00 | 100 ± 0% | 33.33 ± 0.00 | 10.00 ± 0.00 | 0 ± 0% | 0.0 ± 0.0% | 65 ± 0% | 4.12 ± 0.00 | -66.8 ± 0.0 |
| const_clinician | 1 | 7.93 ± 0.00 | 56 ± 0% | 13.26 ± 0.00 | 3.94 ± 0.00 | 60 ± 0% | 2.2 ± 0.0% | 94 ± 0% | 2.82 ± 0.00 | -37.7 ± 0.0 |
| simaan_minflow | 1 | 9.05 ± 0.00 | 65 ± 0% | 15.14 ± 0.00 | 4.49 ± 0.00 | 54 ± 0% | 3.4 ± 0.0% | 93 ± 0% | 2.88 ± 0.00 | -41.2 ± 0.0 |
| oracle_p | 5 | 6.57 ± 0.00 | 43 ± 0% | 15.05 ± 0.92 | 3.27 ± 0.00 | 28 ± 1% | 0.4 ± 0.2% | 92 ± 0% | 2.89 ± 0.00 | -32.7 ± 0.1 |
| oracle | 5 | 6.60 ± 0.00 | 43 ± 0% | 20.18 ± 0.04 | 3.28 ± 0.00 | 28 ± 0% | 0.2 ± 0.0% | 91 ± 0% | 2.91 ± 0.00 | -32.6 ± 0.0 |
| tdm_mlp_v2 | 5 | 7.83 ± 0.20 | 69 ± 5% | 20.64 ± 0.78 | 3.71 ± 0.07 | 32 ± 2% | 0.4 ± 0.1% | 94 ± 1% | 2.89 ± 0.02 | -36.4 ± 0.5 |
| tdm_mlp | 5 | 8.04 ± 0.13 | 63 ± 2% | 21.46 ± 0.64 | 3.89 ± 0.04 | 27 ± 2% | 0.1 ± 0.0% | 93 ± 1% | 2.92 ± 0.01 | -36.8 ± 0.4 |
| ibu_mlp | 5 | 9.28 ± 0.08 | 78 ± 1% | 22.48 ± 0.38 | 4.44 ± 0.04 | 25 ± 1% | 0.1 ± 0.0% | 93 ± 0% | 2.92 ± 0.01 | -40.5 ± 0.2 |
| naive_mlp | 5 | 8.59 ± 0.11 | 65 ± 2% | 20.21 ± 0.28 | 4.14 ± 0.04 | 31 ± 1% | 0.1 ± 0.0% | 95 ± 0% | 2.90 ± 0.00 | -38.5 ± 0.3 |
| rnn | 5 | 7.01 ± 0.18 | 64 ± 9% | 12.53 ± 0.77 | 3.44 ± 0.07 | 60 ± 4% | 1.0 ± 0.6% | 96 ± 1% | 2.77 ± 0.02 | -34.7 ± 0.5 |

## Comparaciones (t de Welch sobre semillas)

| Hipótesis | A − B | Δ violaciones / 100 (p) | Δ gasto (p) | Δ AVO (p) |
| --- | --- | --- | --- | --- |
| H1: RM structure vs memory only | oracle − rnn | -0.401 (0.0839) | +0.135 (0.00145) | -0.318 (0.00107) |
| H2: TDM vs IBU | tdm_mlp − ibu_mlp | -1.242 (0.00013) | +0.001 (0.942) | +0.016 (0.485) |
| H2: TDM vs Naive | tdm_mlp − naive_mlp | -0.548 (0.0137) | +0.020 (0.0796) | -0.046 (0.054) |
| TDM vs memory only | tdm_mlp − rnn | +1.032 (0.0019) | +0.143 (0.000402) | -0.327 (0.000344) |
| H3: pump-only TDM vs RM oracle | tdm_mlp − oracle | +1.434 (0.000432) | +0.008 (0.416) | -0.008 (0.651) |
| H3: pump-only TDM vs pressure oracle | tdm_mlp − oracle_p | +1.468 (0.000393) | +0.022 (0.0589) | -0.015 (0.502) |
| pressure vs pump sensors (oracles) | oracle_p − oracle | -0.035 (1.81e-07) | -0.014 (0.00042) | +0.007 (0.643) |
| TDM v2 vs Phase 4 TDM | tdm_mlp_v2 − tdm_mlp | -0.210 (0.408) | -0.026 (0.322) | +0.048 (0.141) |
| H2: TDM v2 vs IBU | tdm_mlp_v2 − ibu_mlp | -1.452 (0.000827) | -0.025 (0.331) | +0.064 (0.0546) |
| H3: TDM v2 vs RM oracle | tdm_mlp_v2 − oracle | +1.223 (0.0035) | -0.018 (0.461) | +0.040 (0.169) |
