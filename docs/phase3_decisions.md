# Fase 3: agentes, decisiones y punto de decisión

Complementa [results/phase3/summary.md](../results/phase3/summary.md). Una semilla por agente; el experimento principal con 5 semillas es la Fase 5.

## Decisiones tomadas

| Tema | Decisión | Justificación |
| --- | --- | --- |
| Algoritmo | PPO estilo PureJaxRL (Flax y Optax): actor y crítico MLP 2×128 con tanh; versión recurrente con codificador 128, GRU 128 y cabezas separadas | Implementación de referencia del HANDOFF (§13) |
| Hiperparámetros | 8 192 entornos sincronizados, rollouts de 25 latidos (12 por episodio de 300), 4 épocas, minibatches de 8 192 transiciones (recurrente: 32 grupos de 256 entornos con la secuencia completa), lr 3e-4 con decaimiento lineal, γ 0.99, λ 0.95, clip 0.2, entropía 0.01, coeficiente de valor 0.5, recorte de gradiente 0.5; 30 M latidos (~11 min por agente) | Los valores de Li et al. 2024 (Tabla 2), con lotes más grandes porque el entorno está limitado por latencia (48 000 latidos/s con 8 192 entornos) |
| Truncación | El último paso de cada episodio se trata como truncación: se usa el valor de la observación final y no se enmascara | Los episodios de 300 latidos son un límite de tiempo, no un estado terminal |
| Agentes | oracle (bomba + estado de la RM), rnn (GRU, solo bomba), mlp (solo bomba, sin memoria), fz (presiones, ±1%, recompensa r_A + r_B) y fz4 (igual, con ±1% y ±5%) | §9 y §11. En el MDP de Fernández-Zapico la acción 0 siempre obtiene la recompensa máxima, así que la variante de "5 niveles" usa 4 niveles sin el 0 |
| Evaluación | 2 048 escenarios no vistos (semilla 12345), política greedy, todos parten en la velocidad del paciente base (salvo el clínico), métricas con las etiquetas verdaderas y la RM con N = 5; además los 3 protocolos de 1 206 latidos | Misma tubería que las líneas base (`lvadrm.evaluation`) |

## Resultados principales (escenarios no vistos, N = 5)

| Política | Violaciones / 100 latidos | Desde el latido 20 | Episodios con violación desde el latido 20 | AVO | BACK | Gasto medio | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Velocidad constante poblacional | 6.22 | 6.23 | 40.6% | 69% | 21% | 5.11 | −56 |
| Velocidad constante del clínico | 2.80 | 3.00 | 30.3% | 86% | 12% | 5.26 | −35 |
| Simaan | 7.75 | — | — | 61% | 25% | 5.34 | −70 |
| fz / fz4 | 2.95 / 3.15 | — | — | 85% / 83% | 59% / 57% | 4.77 / 4.84 | −106 / −104 |
| mlp | 0.79 | 0.75 | 12.4% | 88% | 0% | 4.84 | −9.5 |
| rnn | 0.79 | 0.68 | 11.9% | 95% | 0% | 4.78 | −9.4 |
| **oracle** | **0.17** | **0.08** | **9.8%** | 68% | 1% | **5.02** | **−9.0** |

El 57% de las violaciones del oracle ocurre en los primeros 20 latidos: el transitorio desde la velocidad inicial común, porque a algunos pacientes les toma más de 5 latidos bajar hasta su velocidad de apertura.

Protocolos, violaciones cada 100 latidos (ejercicio / hipertensión / secuencial):

- **oracle:** 0.0 / 0.4 / 0.0.
- **mlp y rnn:** 0.0 / 15.1–15.2 / 0.0. Fallan en hipertensión.
- **Clínico:** 0.1 / 15.1 / 5.1.
- **Simaan:** 15.2 / 0.0 / 6.6.

## Interpretación

- **Punto de decisión (§14): se supera.** Desde el latido 20, el oracle tiene 37 veces menos violaciones que el clínico (0.08 contra 3.0 cada 100 latidos) y 78 veces menos que la velocidad poblacional, sin succión y con gasto 5.02 l/min. No hace falta simplificar.
- **H1 (estructura de RM contra memoria sola): a favor, con matices.** El oracle tiene 4–9 veces menos violaciones que rnn y mlp, y abre la válvula solo lo necesario (68% de los latidos), lo que le deja más gasto: 5.02 contra 4.78–4.84 l/min, y LOW en 35% contra 41–43%. Pero rnn y mlp llegan a un retorno parecido con una estrategia conservadora, que es mantener la válvula abierta casi siempre (88–95%). Esa estrategia es markoviana, por eso la memoria de rnn casi no supera a mlp. La diferencia clara aparece en hipertensión: ahí no existe velocidad constante que abra la válvula sin regurgitar, y solo el oracle modula la velocidad para abrir periódicamente aceptando BACK (0.4 contra ~15 violaciones cada 100 latidos).
- **MDP de Fernández-Zapico:** con nuestras métricas es malo. Su recompensa no penaliza el reflujo y termina con BACK en 57–59% de los latidos (100% en hipertensión), con más violaciones que el clínico. Para H3 conviene agregar un "oracle con presiones" (presiones + recompensa RM + estado RM) como referencia privilegiada fuerte, además de la réplica.

## Pendiente o por decidir

- Agregar el agente "oracle con presiones" para H3.
- Partir los episodios con la velocidad del clínico (o excluir el transitorio inicial en las métricas principales) para que "episodios con violación" refleje el control y no la inicialización.
- 5 semillas por agente (Fase 5).
