# Sin recorte del valor: ablación del horizonte N (media ± error estándar sobre semillas)

2048 escenarios de prueba no vistos (default), 300 latidos, r_back = -0.1. Política greedy; inicio en la velocidad del clínico (const_pop: velocidad poblacional). Las líneas base son deterministas (n = 1).

| Política | n | Violaciones / 100 latidos (N=5) | Episodios con violación | N=3 | N=10 | AVO | BACK | LOW | Gasto | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | 3 | 0.04 ± 0.01 | 5 ± 2% | 6.99 ± 3.26 | 0.00 ± 0.00 | 67 ± 3% | 0.9 ± 0.2% | 35 ± 0% | 5.05 ± 0.02 | -7.2 ± 0.2 |
| tdm_mlp_v2 | 3 | 0.12 ± 0.00 | 12 ± 1% | 8.23 ± 1.93 | 0.02 ± 0.00 | 70 ± 6% | 1.8 ± 0.1% | 36 ± 1% | 4.98 ± 0.05 | -7.6 ± 0.1 |
| oracle_N3 | 3 | 0.01 ± 0.00 | 0 ± 0% | 0.01 ± 0.00 | 0.00 ± 0.00 | 95 ± 0% | 4.2 ± 0.2% | 40 ± 0% | 4.89 ± 0.01 | -7.7 ± 0.1 |
| tdm_mlp_N3 | 3 | 0.05 ± 0.00 | 4 ± 0% | 0.13 ± 0.01 | 0.01 ± 0.00 | 98 ± 0% | 2.9 ± 0.0% | 40 ± 0% | 4.87 ± 0.01 | -7.7 ± 0.0 |
| oracle_N10 | 3 | 7.97 ± 0.34 | 72 ± 2% | 12.81 ± 1.81 | 0.00 ± 0.00 | 48 ± 2% | 0.3 ± 0.0% | 33 ± 0% | 5.15 ± 0.02 | -7.1 ± 0.1 |
| tdm_mlp_N10 | 3 | 4.30 ± 0.45 | 57 ± 2% | 9.95 ± 0.52 | 0.12 ± 0.01 | 62 ± 2% | 0.9 ± 0.0% | 32 ± 0% | 5.12 ± 0.02 | -7.2 ± 0.1 |

## Comparaciones (t de Welch sobre semillas)

| Hipótesis | A − B | Δ violaciones / 100 (p) | Δ gasto (p) | Δ AVO (p) |
| --- | --- | --- | --- | --- |
| H3: TDM v2 vs RM oracle | tdm_mlp_v2 − oracle | +0.080 (0.017) | -0.066 (0.297) | +0.032 (0.668) |
