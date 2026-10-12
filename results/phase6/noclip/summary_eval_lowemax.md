# Sin recorte del valor: generalización (lowemax) (media ± error estándar sobre semillas)

2048 escenarios de prueba no vistos (lowemax), 300 latidos, r_back = -0.1. Política greedy; inicio en la velocidad del clínico (const_pop: velocidad poblacional). Las líneas base son deterministas (n = 1).

| Política | n | Violaciones / 100 latidos (N=5) | Episodios con violación | N=3 | N=10 | AVO | BACK | LOW | Gasto | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const_pop | 1 | 20.00 ± 0.00 | 100 ± 0% | 33.33 ± 0.00 | 10.00 ± 0.00 | 0 ± 0% | 0.0 ± 0.0% | 65 ± 0% | 4.12 ± 0.00 | -66.8 ± 0.0 |
| const_clinician | 1 | 7.93 ± 0.00 | 56 ± 0% | 13.26 ± 0.00 | 3.94 ± 0.00 | 60 ± 0% | 2.2 ± 0.0% | 94 ± 0% | 2.82 ± 0.00 | -37.7 ± 0.0 |
| simaan_minflow | 1 | 9.05 ± 0.00 | 65 ± 0% | 15.14 ± 0.00 | 4.49 ± 0.00 | 54 ± 0% | 3.4 ± 0.0% | 93 ± 0% | 2.88 ± 0.00 | -41.2 ± 0.0 |
| oracle_p | 5 | 6.58 ± 0.01 | 43 ± 0% | 18.19 ± 0.94 | 3.27 ± 0.00 | 27 ± 0% | 0.3 ± 0.0% | 92 ± 0% | 2.90 ± 0.00 | -32.6 ± 0.0 |
| oracle | 5 | 6.60 ± 0.01 | 44 ± 0% | 18.19 ± 1.37 | 3.28 ± 0.00 | 31 ± 1% | 0.1 ± 0.0% | 94 ± 0% | 2.87 ± 0.01 | -32.7 ± 0.0 |
| tdm_mlp_v2 | 5 | 7.32 ± 0.25 | 61 ± 7% | 18.81 ± 1.50 | 3.54 ± 0.08 | 38 ± 4% | 0.3 ± 0.1% | 95 ± 1% | 2.85 ± 0.02 | -35.1 ± 0.6 |
| tdm_mlp | 5 | 7.86 ± 0.05 | 59 ± 1% | 20.68 ± 0.38 | 3.85 ± 0.02 | 29 ± 1% | 0.0 ± 0.0% | 94 ± 0% | 2.90 ± 0.00 | -36.3 ± 0.1 |
| ibu_mlp | 5 | 8.39 ± 0.14 | 69 ± 2% | 20.03 ± 0.40 | 4.04 ± 0.05 | 33 ± 1% | 0.3 ± 0.0% | 95 ± 1% | 2.87 ± 0.01 | -38.1 ± 0.4 |
| naive_mlp | 5 | 8.34 ± 0.10 | 61 ± 0% | 19.29 ± 0.20 | 4.06 ± 0.04 | 35 ± 1% | 0.1 ± 0.0% | 95 ± 0% | 2.88 ± 0.00 | -37.9 ± 0.3 |
| rnn | 5 | 6.89 ± 0.05 | 55 ± 4% | 12.47 ± 0.57 | 3.40 ± 0.03 | 59 ± 3% | 0.5 ± 0.2% | 96 ± 1% | 2.77 ± 0.02 | -34.3 ± 0.2 |

## Comparaciones (t de Welch sobre semillas)

| Hipótesis | A − B | Δ violaciones / 100 (p) | Δ gasto (p) | Δ AVO (p) |
| --- | --- | --- | --- | --- |
| H1: RM structure vs memory only | oracle − rnn | -0.297 (0.00494) | +0.104 (0.00639) | -0.281 (0.000608) |
| H2: TDM vs IBU | tdm_mlp − ibu_mlp | -0.533 (0.0162) | +0.023 (0.088) | -0.041 (0.0416) |
| H2: TDM vs Naive | tdm_mlp − naive_mlp | -0.483 (0.00568) | +0.020 (0.00666) | -0.053 (0.00275) |
| TDM vs memory only | tdm_mlp − rnn | +0.965 (1.05e-06) | +0.130 (0.00293) | -0.300 (0.000442) |
| H3: pump-only TDM vs RM oracle | tdm_mlp − oracle | +1.262 (7.94e-06) | +0.025 (0.00661) | -0.019 (0.201) |
| H3: pump-only TDM vs pressure oracle | tdm_mlp − oracle_p | +1.275 (6.71e-06) | -0.007 (0.341) | +0.025 (0.0578) |
| pressure vs pump sensors (oracles) | oracle_p − oracle | -0.013 (0.263) | +0.032 (0.00213) | -0.044 (0.00741) |
| TDM v2 vs Phase 4 TDM | tdm_mlp_v2 − tdm_mlp | -0.534 (0.103) | -0.052 (0.0865) | +0.084 (0.119) |
| H2: TDM v2 vs IBU | tdm_mlp_v2 − ibu_mlp | -1.067 (0.00959) | -0.029 (0.298) | +0.043 (0.383) |
| H3: TDM v2 vs RM oracle | tdm_mlp_v2 − oracle | +0.729 (0.0452) | -0.027 (0.314) | +0.065 (0.202) |
