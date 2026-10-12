# Sin recorte del valor: experimento principal (Fases 5 y 6A) (media ± error estándar sobre semillas)

2048 escenarios de prueba no vistos (default), 300 latidos, r_back = -0.1. Política greedy; inicio en la velocidad del clínico (const_pop: velocidad poblacional). Las líneas base son deterministas (n = 1).

| Política | n | Violaciones / 100 latidos (N=5) | Episodios con violación | N=3 | N=10 | AVO | BACK | LOW | Gasto | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const_pop | 1 | 6.22 ± 0.00 | 41 ± 0% | 10.39 ± 0.00 | 3.10 ± 0.00 | 69 ± 0% | 20.6 ± 0.0% | 38 ± 0% | 5.11 ± 0.00 | -31.5 ± 0.0 |
| const_clinician | 1 | 2.80 ± 0.00 | 30 ± 0% | 4.71 ± 0.00 | 1.38 ± 0.00 | 86 ± 0% | 11.8 ± 0.0% | 38 ± 0% | 5.26 ± 0.00 | -20.3 ± 0.0 |
| simaan_minflow | 1 | 4.38 ± 0.00 | 45 ± 0% | 7.35 ± 0.00 | 2.15 ± 0.00 | 78 ± 0% | 19.6 ± 0.0% | 37 ± 0% | 5.47 ± 0.00 | -30.2 ± 0.0 |
| mod_open_loop | 1 | 0.96 ± 0.00 | 12 ± 0% | 3.46 ± 0.00 | 0.47 ± 0.00 | 89 ± 0% | 19.2 ± 0.0% | 39 ± 0% | 5.17 ± 0.00 | -17.4 ± 0.0 |
| mod_rule_mlp | 1 | 0.65 ± 0.00 | 27 ± 0% | 1.27 ± 0.00 | 0.24 ± 0.00 | 95 ± 0% | 5.9 ± 0.0% | 37 ± 0% | 5.26 ± 0.00 | -12.2 ± 0.0 |
| mod_rule_oracle | 1 | 0.01 ± 0.00 | 0 ± 0% | 0.08 ± 0.00 | 0.00 ± 0.00 | 93 ± 0% | 4.0 ± 0.0% | 37 ± 0% | 5.32 ± 0.00 | -9.9 ± 0.0 |
| oracle_p | 5 | 0.01 ± 0.00 | 0 ± 0% | 9.14 ± 1.46 | 0.00 ± 0.00 | 52 ± 2% | 6.0 ± 1.4% | 35 ± 0% | 5.23 ± 0.03 | -9.8 ± 0.3 |
| oracle | 5 | 0.03 ± 0.01 | 3 ± 2% | 5.65 ± 2.00 | 0.00 ± 0.00 | 71 ± 3% | 1.3 ± 0.3% | 35 ± 0% | 5.04 ± 0.01 | -7.2 ± 0.1 |
| tdm_mlp_v2 | 5 | 0.13 ± 0.01 | 14 ± 2% | 8.32 ± 1.33 | 0.03 ± 0.00 | 70 ± 4% | 1.8 ± 0.1% | 36 ± 1% | 4.99 ± 0.03 | -7.6 ± 0.1 |
| tdm_mlp | 5 | 0.15 ± 0.02 | 12 ± 0% | 10.74 ± 0.13 | 0.04 ± 0.01 | 59 ± 1% | 1.4 ± 0.1% | 35 ± 0% | 5.06 ± 0.01 | -7.7 ± 0.0 |
| ibu_mlp | 5 | 0.19 ± 0.01 | 14 ± 1% | 3.26 ± 0.13 | 0.04 ± 0.00 | 85 ± 1% | 1.5 ± 0.1% | 38 ± 0% | 4.93 ± 0.01 | -7.5 ± 0.0 |
| naive_mlp | 5 | 0.26 ± 0.02 | 20 ± 1% | 7.06 ± 0.38 | 0.07 ± 0.01 | 71 ± 1% | 1.9 ± 0.1% | 36 ± 0% | 4.97 ± 0.01 | -8.0 ± 0.0 |
| rnn | 5 | 0.35 ± 0.08 | 18 ± 3% | 1.79 ± 0.80 | 0.13 ± 0.04 | 92 ± 3% | 1.8 ± 0.4% | 39 ± 1% | 4.90 ± 0.05 | -8.2 ± 0.2 |

## Protocolos (1206 latidos): violaciones / 100 latidos (N = 5)

| Política | ferreira_exercise | ferreira_hypertension | fz_sequential |
| --- | --- | --- | --- |
| const_pop | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.82 ± 0.00 |
| const_clinician | 0.08 ± 0.00 | 15.09 ± 0.00 | 5.06 ± 0.00 |
| simaan_minflow | 15.59 ± 0.00 | 0.00 ± 0.00 | 7.38 ± 0.00 |
| mod_open_loop | 0.00 ± 0.00 | 0.00 ± 0.00 | 2.07 ± 0.00 |
| mod_rule_mlp | 0.00 ± 0.00 | 14.34 ± 0.00 | 0.83 ± 0.00 |
| mod_rule_oracle | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| oracle_p | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| oracle | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| tdm_mlp_v2 | 0.00 ± 0.00 | 5.36 ± 1.56 | 0.10 ± 0.10 |
| tdm_mlp | 0.00 ± 0.00 | 14.81 ± 0.04 | 0.25 ± 0.08 |
| ibu_mlp | 0.00 ± 0.00 | 14.96 ± 0.03 | 0.00 ± 0.00 |
| naive_mlp | 0.00 ± 0.00 | 14.21 ± 0.09 | 0.08 ± 0.05 |
| rnn | 0.00 ± 0.00 | 3.48 ± 1.85 | 0.17 ± 0.07 |

## Comparaciones (t de Welch sobre semillas)

| Hipótesis | A − B | Δ violaciones / 100 (p) | Δ gasto (p) | Δ AVO (p) |
| --- | --- | --- | --- | --- |
| H1: RM structure vs memory only | oracle − rnn | -0.326 (0.0155) | +0.131 (0.063) | -0.213 (0.00135) |
| H2: TDM vs IBU | tdm_mlp − ibu_mlp | -0.037 (0.0887) | +0.129 (2.28e-05) | -0.264 (1.1e-08) |
| H2: TDM vs Naive | tdm_mlp − naive_mlp | -0.114 (0.00217) | +0.093 (0.000191) | -0.122 (0.000236) |
| TDM vs memory only | tdm_mlp − rnn | -0.202 (0.0668) | +0.157 (0.0379) | -0.333 (0.000365) |
| H3: pump-only TDM vs RM oracle | tdm_mlp − oracle | +0.123 (0.000379) | +0.025 (0.18) | -0.120 (0.0161) |
| H3: pump-only TDM vs pressure oracle | tdm_mlp − oracle_p | +0.143 (0.000879) | -0.170 (0.00265) | +0.068 (0.0285) |
| pressure vs pump sensors (oracles) | oracle_p − oracle | -0.020 (0.142) | +0.196 (0.000978) | -0.187 (0.00143) |
| TDM v2 vs Phase 4 TDM | tdm_mlp_v2 − tdm_mlp | -0.017 (0.434) | -0.072 (0.0928) | +0.111 (0.0447) |
| H2: TDM v2 vs IBU | tdm_mlp_v2 − ibu_mlp | -0.053 (0.00962) | +0.058 (0.157) | -0.153 (0.0153) |
| H3: TDM v2 vs RM oracle | tdm_mlp_v2 − oracle | +0.106 (0.000239) | -0.046 (0.246) | -0.009 (0.857) |
