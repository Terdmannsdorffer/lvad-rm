# Fase 6B: generalización (wide) (media ± error estándar sobre semillas)

2048 escenarios de prueba no vistos (wide), 300 latidos, r_back = -0.1. Política greedy; inicio en la velocidad del clínico (const_pop: velocidad poblacional). Las líneas base son deterministas (n = 1).

| Política | n | Violaciones / 100 latidos (N=5) | Episodios con violación | N=3 | N=10 | AVO | BACK | LOW | Gasto | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const_pop | 1 | 8.21 ± 0.00 | 51 ± 0% | 13.71 ± 0.00 | 4.09 ± 0.00 | 59 ± 0% | 49.3 ± 0.0% | 73 ± 0% | 3.75 ± 0.00 | -48.6 ± 0.0 |
| const_clinician | 1 | 2.87 ± 0.00 | 31 ± 0% | 4.82 ± 0.00 | 1.41 ± 0.00 | 85 ± 0% | 55.3 ± 0.0% | 73 ± 0% | 3.74 ± 0.00 | -35.3 ± 0.0 |
| simaan_minflow | 1 | 4.51 ± 0.00 | 45 ± 0% | 7.57 ± 0.00 | 2.22 ± 0.00 | 77 ± 0% | 54.0 ± 0.0% | 71 ± 0% | 3.89 ± 0.00 | -41.9 ± 0.0 |
| oracle_p | 5 | 0.01 ± 0.00 | 0 ± 0% | 6.87 ± 1.73 | 0.00 ± 0.00 | 47 ± 3% | 49.0 ± 1.3% | 70 ± 0% | 3.79 ± 0.01 | -24.6 ± 0.4 |
| oracle | 5 | 0.12 ± 0.11 | 5 ± 4% | 15.14 ± 0.21 | 0.03 ± 0.03 | 45 ± 1% | 42.9 ± 0.0% | 68 ± 0% | 3.82 ± 0.01 | -22.9 ± 0.3 |
| tdm_mlp_v2 | 5 | 6.81 ± 0.24 | 64 ± 1% | 19.44 ± 0.56 | 3.03 ± 0.14 | 36 ± 1% | 40.8 ± 0.3% | 69 ± 0% | 3.81 ± 0.02 | -42.0 ± 0.7 |
| tdm_mlp | 5 | 8.58 ± 0.19 | 63 ± 1% | 21.36 ± 0.78 | 4.10 ± 0.08 | 29 ± 3% | 34.9 ± 0.5% | 69 ± 1% | 3.86 ± 0.02 | -45.4 ± 0.6 |
| ibu_mlp | 5 | 8.99 ± 0.07 | 65 ± 0% | 19.19 ± 0.34 | 4.28 ± 0.03 | 36 ± 2% | 35.5 ± 0.5% | 70 ± 0% | 3.81 ± 0.01 | -46.6 ± 0.4 |
| naive_mlp | 5 | 8.69 ± 0.03 | 67 ± 0% | 20.56 ± 0.20 | 4.15 ± 0.01 | 31 ± 1% | 33.4 ± 0.1% | 69 ± 0% | 3.84 ± 0.00 | -45.2 ± 0.1 |
| rnn | 5 | 6.08 ± 1.02 | 66 ± 2% | 13.72 ± 0.92 | 2.60 ± 0.53 | 52 ± 2% | 39.2 ± 1.9% | 71 ± 1% | 3.76 ± 0.02 | -39.3 ± 2.9 |

## Comparaciones (t de Welch sobre semillas)

| Hipótesis | A − B | Δ violaciones / 100 (p) | Δ gasto (p) | Δ AVO (p) |
| --- | --- | --- | --- | --- |
| H1: RM structure vs memory only | oracle − rnn | -5.967 (0.00408) | +0.057 (0.0399) | -0.073 (0.0255) |
| H2: TDM vs IBU | tdm_mlp − ibu_mlp | -0.408 (0.0961) | +0.043 (0.0687) | -0.077 (0.0625) |
| H2: TDM vs Naive | tdm_mlp − naive_mlp | -0.105 (0.609) | +0.019 (0.348) | -0.027 (0.424) |
| TDM vs memory only | tdm_mlp − rnn | +2.497 (0.0698) | +0.095 (0.00657) | -0.237 (0.000311) |
| H3: pump-only TDM vs RM oracle | tdm_mlp − oracle | +8.464 (8.47e-09) | +0.039 (0.0925) | -0.164 (0.00407) |
| H3: pump-only TDM vs pressure oracle | tdm_mlp − oracle_p | +8.575 (1.37e-06) | +0.071 (0.0137) | -0.182 (0.00358) |
| pressure vs pump sensors (oracles) | oracle_p − oracle | -0.111 (0.357) | -0.032 (0.0739) | +0.018 (0.622) |
| TDM v2 vs Phase 4 TDM | tdm_mlp_v2 − tdm_mlp | -1.768 (0.000535) | -0.049 (0.0755) | +0.077 (0.0627) |
| H2: TDM v2 vs IBU | tdm_mlp_v2 − ibu_mlp | -2.176 (0.000513) | -0.006 (0.744) | -0.001 (0.972) |
| H3: TDM v2 vs RM oracle | tdm_mlp_v2 − oracle | +6.696 (7.07e-07) | -0.011 (0.573) | -0.087 (0.00112) |
