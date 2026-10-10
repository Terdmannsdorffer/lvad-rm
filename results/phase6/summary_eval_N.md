# Fase 6D: ablación del horizonte N (media ± error estándar sobre semillas)

2048 escenarios de prueba no vistos (default), 300 latidos, r_back = -0.1. Política greedy; inicio en la velocidad del clínico (const_pop: velocidad poblacional). Las líneas base son deterministas (n = 1).

| Política | n | Violaciones / 100 latidos (N=5) | Episodios con violación | N=3 | N=10 | AVO | BACK | LOW | Gasto | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | 3 | 0.01 ± 0.00 | 0 ± 0% | 10.67 ± 0.26 | 0.00 ± 0.00 | 60 ± 1% | 2.0 ± 0.0% | 33 ± 0% | 5.14 ± 0.00 | -7.8 ± 0.0 |
| tdm_mlp_v2 | 3 | 0.17 ± 0.02 | 16 ± 1% | 10.04 ± 1.15 | 0.03 ± 0.01 | 66 ± 3% | 2.2 ± 0.2% | 35 ± 1% | 5.05 ± 0.04 | -8.0 ± 0.1 |
| oracle_N3 | 3 | 6.54 ± 6.53 | 34 ± 33% | 10.97 ± 10.95 | 3.21 ± 3.20 | 58 ± 29% | 2.9 ± 1.7% | 25 ± 12% | 5.77 ± 0.78 | -43.0 ± 34.9 |
| tdm_mlp_N3 | 3 | 6.59 ± 6.54 | 36 ± 32% | 11.10 ± 10.95 | 3.23 ± 3.22 | 65 ± 32% | 1.9 ± 1.0% | 26 ± 13% | 6.45 ± 1.55 | -47.3 ± 39.5 |
| oracle_N10 | 3 | 6.77 ± 0.51 | 65 ± 2% | 12.92 ± 0.56 | 0.01 ± 0.00 | 51 ± 0% | 0.3 ± 0.0% | 32 ± 1% | 5.14 ± 0.02 | -6.9 ± 0.0 |
| tdm_mlp_N10 | 3 | 4.54 ± 1.56 | 58 ± 8% | 10.38 ± 1.30 | 0.15 ± 0.02 | 61 ± 5% | 0.9 ± 0.0% | 32 ± 1% | 5.13 ± 0.05 | -7.4 ± 0.2 |

## Comparaciones (t de Welch sobre semillas)

| Hipótesis | A − B | Δ violaciones / 100 (p) | Δ gasto (p) | Δ AVO (p) |
| --- | --- | --- | --- | --- |
| H3: TDM v2 vs RM oracle | tdm_mlp_v2 − oracle | +0.158 (0.0124) | -0.096 (0.12) | +0.060 (0.195) |
