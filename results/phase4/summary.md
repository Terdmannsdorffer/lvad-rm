# Fase 4: agentes PPO contra líneas base

2048 escenarios de prueba no vistos (semilla 12345), 300 latidos, RM con N = 5. Agentes con política greedy. Todas las políticas parten en la velocidad del clínico, salvo const_pop (velocidad poblacional).

| Política | Episodios con violación (N=5) | Violaciones / 100 latidos | Ídem desde el latido 20 | AVO | BACK | SUC | LOW | Gasto medio | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const_pop | 41% | 6.22 | 41%, 6.23 | 69% | 21% | 0.00% | 38% | 5.11 | -56.2 ± 50.3 |
| const_clinician | 30% | 2.80 | 30%, 3.00 | 86% | 12% | 0.00% | 38% | 5.26 | -34.5 ± 42.3 |
| simaan_minflow | 45% | 4.38 | 45%, 4.69 | 78% | 20% | 0.23% | 37% | 5.47 | -53.7 ± 49.5 |
| oracle_s0 | 10% | 0.07 | 10%, 0.08 | 68% | 1% | 0.00% | 35% | 5.02 | -8.5 ± 6.7 |
| oracle_p_s0 | 0% | 0.01 | 0%, 0.01 | 47% | 0% | 0.00% | 35% | 5.29 | -8.8 ± 7.4 |
| rnn_s0 | 20% | 0.50 | 10%, 0.41 | 97% | 0% | 0.00% | 43% | 4.78 | -8.6 ± 9.8 |
| naive_mlp_s0 | 18% | 0.64 | 17%, 0.65 | 87% | 0% | 0.00% | 39% | 4.90 | -8.9 ± 9.2 |
| ibu_mlp_s0 | 17% | 0.62 | 17%, 0.64 | 85% | 0% | 0.00% | 38% | 4.91 | -8.9 ± 10.2 |
| tdm_mlp_s0 | 14% | 0.30 | 13%, 0.29 | 80% | 1% | 0.00% | 38% | 4.93 | -8.4 ± 8.0 |
| naive_logistic_s0 | 10% | 0.59 | 10%, 0.61 | 89% | 0% | 0.00% | 42% | 4.83 | -8.4 ± 9.4 |
| ibu_logistic_s0 | 10% | 0.56 | 10%, 0.58 | 88% | 0% | 0.00% | 41% | 4.83 | -8.4 ± 9.2 |
| tdm_logistic_s0 | 17% | 0.49 | 17%, 0.52 | 74% | 1% | 0.00% | 37% | 4.98 | -9.5 ± 9.6 |
| naive_markov_e0.2_r0.9_s0 | 14% | 0.38 | 14%, 0.40 | 90% | 0% | 0.00% | 42% | 4.82 | -8.0 ± 7.0 |
| ibu_markov_e0.2_r0.9_s0 | 16% | 0.35 | 16%, 0.37 | 89% | 0% | 0.00% | 42% | 4.82 | -7.9 ± 6.6 |
| tdm_det_markov_e0.2_r0.9_s0 | 12% | 0.48 | 12%, 0.48 | 91% | 0% | 0.00% | 41% | 4.82 | -8.3 ± 8.8 |
| tdm_markov_e0.2_r0.9_s0 | 21% | 0.20 | 19%, 0.20 | 70% | 1% | 0.00% | 37% | 4.98 | -9.0 ± 7.2 |

## Protocolos (1206 latidos): violaciones / 100 latidos (N = 5), BACK y gasto

| Política | ferreira_exercise | ferreira_hypertension | fz_sequential |
| --- | --- | --- | --- |
| const_pop | 0.0 (BACK 0%, gasto 6.6) | 0.0 (BACK 78%, gasto 3.6) | 1.8 (BACK 14%, gasto 5.0) |
| const_clinician | 0.1 (BACK 0%, gasto 6.8) | 15.1 (BACK 75%, gasto 3.8) | 5.1 (BACK 0%, gasto 5.3) |
| simaan_minflow | 15.6 (BACK 0%, gasto 9.0) | 0.0 (BACK 78%, gasto 3.2) | 7.4 (BACK 39%, gasto 5.5) |
| oracle_s0 | 0.0 (BACK 0%, gasto 6.4) | 0.8 (BACK 60%, gasto 3.8) | 0.0 (BACK 0%, gasto 5.2) |
| oracle_p_s0 | 0.0 (BACK 0%, gasto 6.8) | 0.0 (BACK 75%, gasto 3.8) | 0.0 (BACK 0%, gasto 5.4) |
| rnn_s0 | 0.0 (BACK 0%, gasto 6.2) | 15.2 (BACK 75%, gasto 3.8) | 0.0 (BACK 0%, gasto 4.9) |
| naive_mlp_s0 | 0.0 (BACK 0%, gasto 6.3) | 15.1 (BACK 75%, gasto 3.8) | 0.0 (BACK 0%, gasto 5.0) |
| ibu_mlp_s0 | 0.0 (BACK 0%, gasto 6.3) | 15.0 (BACK 57%, gasto 3.8) | 0.0 (BACK 0%, gasto 5.0) |
| tdm_mlp_s0 | 0.0 (BACK 0%, gasto 6.3) | 15.0 (BACK 55%, gasto 3.8) | 0.0 (BACK 0%, gasto 5.1) |
| naive_logistic_s0 | 0.0 (BACK 0%, gasto 6.3) | 15.1 (BACK 71%, gasto 3.8) | 0.0 (BACK 0%, gasto 4.9) |
| ibu_logistic_s0 | 0.0 (BACK 0%, gasto 6.3) | 15.1 (BACK 67%, gasto 3.8) | 0.0 (BACK 0%, gasto 4.9) |
| tdm_logistic_s0 | 0.0 (BACK 0%, gasto 6.4) | 10.8 (BACK 53%, gasto 3.8) | 0.0 (BACK 0%, gasto 5.1) |
| naive_markov_e0.2_r0.9_s0 | 0.0 (BACK 0%, gasto 6.2) | 15.2 (BACK 46%, gasto 3.8) | 0.0 (BACK 0%, gasto 4.9) |
| ibu_markov_e0.2_r0.9_s0 | 0.0 (BACK 0%, gasto 6.2) | 15.1 (BACK 35%, gasto 3.8) | 0.0 (BACK 1%, gasto 4.9) |
| tdm_det_markov_e0.2_r0.9_s0 | 0.0 (BACK 0%, gasto 6.2) | 15.1 (BACK 38%, gasto 3.9) | 0.0 (BACK 0%, gasto 4.9) |
| tdm_markov_e0.2_r0.9_s0 | 0.0 (BACK 0%, gasto 6.3) | 5.6 (BACK 58%, gasto 3.8) | 0.0 (BACK 0%, gasto 5.1) |
