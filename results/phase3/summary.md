# Fase 3: agentes PPO contra líneas base

2048 escenarios de prueba no vistos (semilla 12345), 300 latidos, RM con N = 5. Agentes con política greedy. Todas las políticas parten en la velocidad del clínico, salvo const_pop (velocidad poblacional).

| Política | Episodios con violación (N=5) | Violaciones / 100 latidos | Ídem desde el latido 20 | AVO | BACK | SUC | LOW | Gasto medio | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const_pop | 41% | 6.22 | 41%, 6.23 | 69% | 21% | 0.00% | 38% | 5.11 | -56.2 ± 50.3 |
| const_clinician | 30% | 2.80 | 30%, 3.00 | 86% | 12% | 0.00% | 38% | 5.26 | -34.5 ± 42.3 |
| simaan_minflow | 45% | 4.38 | 45%, 4.69 | 78% | 20% | 0.23% | 37% | 5.47 | -53.7 ± 49.5 |
| fz4_s0 | 23% | 3.03 | 23%, 3.07 | 83% | 57% | 0.00% | 51% | 4.84 | -103.6 ± 55.3 |
| fz_s0 | 22% | 2.87 | 22%, 2.91 | 85% | 59% | 0.00% | 52% | 4.78 | -105.6 ± 55.7 |
| mlp_s0 | 12% | 0.73 | 12%, 0.76 | 88% | 0% | 0.00% | 41% | 4.84 | -8.9 ± 10.5 |
| oracle_p_s0 | 0% | 0.01 | 0%, 0.01 | 47% | 0% | 0.00% | 35% | 5.29 | -8.8 ± 7.4 |
| oracle_s0 | 10% | 0.07 | 10%, 0.08 | 68% | 1% | 0.00% | 35% | 5.02 | -8.5 ± 6.7 |
| rnn_s0 | 20% | 0.50 | 10%, 0.41 | 97% | 0% | 0.00% | 43% | 4.78 | -8.6 ± 9.8 |

## Protocolos (1206 latidos): violaciones / 100 latidos (N = 5), BACK y gasto

| Política | ferreira_exercise | ferreira_hypertension | fz_sequential |
| --- | --- | --- | --- |
| const_pop | 0.0 (BACK 0%, gasto 6.6) | 0.0 (BACK 78%, gasto 3.6) | 1.8 (BACK 14%, gasto 5.0) |
| const_clinician | 0.1 (BACK 0%, gasto 6.8) | 15.1 (BACK 75%, gasto 3.8) | 5.1 (BACK 0%, gasto 5.3) |
| simaan_minflow | 15.6 (BACK 0%, gasto 9.0) | 0.0 (BACK 78%, gasto 3.2) | 7.4 (BACK 39%, gasto 5.5) |
| fz4_s0 | 0.0 (BACK 86%, gasto 6.0) | 0.0 (BACK 97%, gasto 3.2) | 0.5 (BACK 76%, gasto 4.6) |
| fz_s0 | 0.0 (BACK 99%, gasto 6.0) | 0.0 (BACK 99%, gasto 3.1) | 0.5 (BACK 78%, gasto 4.6) |
| mlp_s0 | 0.0 (BACK 0%, gasto 6.2) | 15.1 (BACK 65%, gasto 3.8) | 0.0 (BACK 0%, gasto 4.9) |
| oracle_p_s0 | 0.0 (BACK 0%, gasto 6.8) | 0.0 (BACK 75%, gasto 3.8) | 0.0 (BACK 0%, gasto 5.4) |
| oracle_s0 | 0.0 (BACK 0%, gasto 6.4) | 0.8 (BACK 60%, gasto 3.8) | 0.0 (BACK 0%, gasto 5.2) |
| rnn_s0 | 0.0 (BACK 0%, gasto 6.2) | 15.2 (BACK 75%, gasto 3.8) | 0.0 (BACK 0%, gasto 4.9) |
