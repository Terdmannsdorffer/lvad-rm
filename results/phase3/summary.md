# Fase 3: agentes PPO contra líneas base

2048 escenarios de prueba no vistos (semilla 12345), 300 latidos, RM con N = 5. Agentes con política greedy.

| Política | Episodios con violación (N=5) | Violaciones / 100 latidos | AVO | BACK | SUC | LOW | Gasto medio | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const_pop | 41% | 6.22 | 69% | 21% | 0.00% | 38% | 5.11 | -56.2 ± 50.3 |
| const_clinician | 30% | 2.80 | 86% | 12% | 0.00% | 38% | 5.26 | -34.5 ± 42.3 |
| simaan_minflow | 60% | 7.75 | 61% | 25% | 0.19% | 37% | 5.34 | -69.6 ± 49.3 |
| fz4_s0 | 36% | 3.15 | 83% | 57% | 0.00% | 51% | 4.84 | -103.9 ± 55.1 |
| fz_s0 | 35% | 2.95 | 85% | 59% | 0.00% | 52% | 4.77 | -105.9 ± 55.6 |
| mlp_s0 | 20% | 0.79 | 88% | 0% | 0.00% | 41% | 4.84 | -9.5 ± 10.6 |
| oracle_s0 | 24% | 0.17 | 68% | 1% | 0.00% | 35% | 5.02 | -9.0 ± 7.0 |
| rnn_s0 | 28% | 0.79 | 95% | 0% | 0.00% | 43% | 4.78 | -9.4 ± 10.5 |

## Protocolos (1206 latidos): violaciones / 100 latidos (N = 5), BACK y gasto

| Política | ferreira_exercise | ferreira_hypertension | fz_sequential |
| --- | --- | --- | --- |
| const_pop | 0.0 (BACK 0%, gasto 6.6) | 0.0 (BACK 78%, gasto 3.6) | 1.8 (BACK 14%, gasto 5.0) |
| const_clinician | 0.1 (BACK 0%, gasto 6.8) | 15.1 (BACK 75%, gasto 3.8) | 5.1 (BACK 0%, gasto 5.3) |
| simaan_minflow | 15.2 (BACK 0%, gasto 8.7) | 0.0 (BACK 79%, gasto 3.1) | 6.6 (BACK 50%, gasto 5.4) |
| fz4_s0 | 0.0 (BACK 87%, gasto 6.0) | 0.0 (BACK 98%, gasto 3.2) | 0.5 (BACK 77%, gasto 4.6) |
| fz_s0 | 0.0 (BACK 100%, gasto 6.0) | 0.0 (BACK 100%, gasto 3.1) | 0.5 (BACK 79%, gasto 4.6) |
| mlp_s0 | 0.0 (BACK 0%, gasto 6.3) | 15.1 (BACK 65%, gasto 3.8) | 0.0 (BACK 0%, gasto 4.9) |
| oracle_s0 | 0.0 (BACK 0%, gasto 6.4) | 0.4 (BACK 59%, gasto 3.8) | 0.0 (BACK 0%, gasto 5.2) |
| rnn_s0 | 0.0 (BACK 0%, gasto 6.2) | 15.2 (BACK 75%, gasto 3.8) | 0.0 (BACK 0%, gasto 4.9) |
