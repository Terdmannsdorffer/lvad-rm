# Sin recorte del valor: generalización (wide) (media ± error estándar sobre semillas)

2048 escenarios de prueba no vistos (wide), 300 latidos, r_back = -0.1. Política greedy; inicio en la velocidad del clínico (const_pop: velocidad poblacional). Las líneas base son deterministas (n = 1).

| Política | n | Violaciones / 100 latidos (N=5) | Episodios con violación | N=3 | N=10 | AVO | BACK | LOW | Gasto | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const_pop | 1 | 8.21 ± 0.00 | 51 ± 0% | 13.71 ± 0.00 | 4.09 ± 0.00 | 59 ± 0% | 49.3 ± 0.0% | 73 ± 0% | 3.75 ± 0.00 | -48.6 ± 0.0 |
| const_clinician | 1 | 2.87 ± 0.00 | 31 ± 0% | 4.82 ± 0.00 | 1.41 ± 0.00 | 85 ± 0% | 55.3 ± 0.0% | 73 ± 0% | 3.74 ± 0.00 | -35.3 ± 0.0 |
| simaan_minflow | 1 | 4.51 ± 0.00 | 45 ± 0% | 7.57 ± 0.00 | 2.22 ± 0.00 | 77 ± 0% | 54.0 ± 0.0% | 71 ± 0% | 3.89 ± 0.00 | -41.9 ± 0.0 |
| oracle_p | 5 | 0.01 ± 0.00 | 0 ± 0% | 10.19 ± 1.67 | 0.00 ± 0.00 | 43 ± 1% | 46.9 ± 1.0% | 69 ± 0% | 3.81 ± 0.01 | -24.0 ± 0.3 |
| oracle | 5 | 0.80 ± 0.36 | 19 ± 5% | 11.60 ± 2.83 | 0.28 ± 0.16 | 49 ± 3% | 44.7 ± 0.8% | 70 ± 0% | 3.77 ± 0.01 | -25.3 ± 0.9 |
| tdm_mlp_v2 | 5 | 6.66 ± 0.19 | 62 ± 1% | 18.95 ± 0.75 | 2.99 ± 0.10 | 37 ± 2% | 41.7 ± 0.4% | 70 ± 0% | 3.79 ± 0.02 | -41.9 ± 0.6 |
| tdm_mlp | 5 | 8.51 ± 0.05 | 63 ± 0% | 22.11 ± 0.05 | 4.08 ± 0.03 | 27 ± 0% | 35.2 ± 0.1% | 69 ± 0% | 3.86 ± 0.00 | -45.3 ± 0.2 |
| ibu_mlp | 5 | 9.26 ± 0.05 | 64 ± 0% | 17.96 ± 0.06 | 4.40 ± 0.02 | 42 ± 0% | 39.6 ± 0.2% | 72 ± 0% | 3.77 ± 0.00 | -48.8 ± 0.2 |
| naive_mlp | 5 | 8.85 ± 0.11 | 67 ± 0% | 20.13 ± 0.19 | 4.23 ± 0.05 | 33 ± 1% | 34.5 ± 0.5% | 70 ± 0% | 3.82 ± 0.01 | -46.0 ± 0.5 |
| rnn | 5 | 5.38 ± 1.06 | 63 ± 4% | 11.35 ± 1.23 | 2.39 ± 0.51 | 60 ± 3% | 45.4 ± 0.5% | 72 ± 1% | 3.71 ± 0.02 | -39.3 ± 3.1 |

## Comparaciones (t de Welch sobre semillas)

| Hipótesis | A − B | Δ violaciones / 100 (p) | Δ gasto (p) | Δ AVO (p) |
| --- | --- | --- | --- | --- |
| H1: RM structure vs memory only | oracle − rnn | -4.586 (0.00978) | +0.055 (0.0674) | -0.109 (0.052) |
| H2: TDM vs IBU | tdm_mlp − ibu_mlp | -0.753 (5.95e-06) | +0.083 (1.14e-06) | -0.155 (7.45e-09) |
| H2: TDM vs Naive | tdm_mlp − naive_mlp | -0.340 (0.0346) | +0.038 (0.00259) | -0.067 (0.000356) |
| TDM vs memory only | tdm_mlp − rnn | +3.126 (0.0421) | +0.145 (0.00193) | -0.334 (0.000556) |
| H3: pump-only TDM vs RM oracle | tdm_mlp − oracle | +7.712 (2.15e-05) | +0.090 (0.00183) | -0.225 (0.00258) |
| H3: pump-only TDM vs pressure oracle | tdm_mlp − oracle_p | +8.502 (8.96e-09) | +0.047 (0.0235) | -0.164 (9.58e-05) |
| pressure vs pump sensors (oracles) | oracle_p − oracle | -0.790 (0.0946) | +0.043 (0.0573) | -0.061 (0.153) |
| TDM v2 vs Phase 4 TDM | tdm_mlp_v2 − tdm_mlp | -1.852 (0.000353) | -0.070 (0.0141) | +0.102 (0.0122) |
| H2: TDM v2 vs IBU | tdm_mlp_v2 − ibu_mlp | -2.605 (8.86e-05) | +0.012 (0.529) | -0.052 (0.0933) |
| H3: TDM v2 vs RM oracle | tdm_mlp_v2 − oracle | +5.860 (6.78e-06) | +0.019 (0.415) | -0.122 (0.0207) |
