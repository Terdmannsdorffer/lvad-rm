# Fase 3: agentes, decisiones y punto de decisión

Complementa [results/phase3/summary.md](../results/phase3/summary.md). Una semilla por agente; el experimento principal con 5 semillas es la Fase 5.

## Decisiones tomadas

| Tema | Decisión | Justificación |
| --- | --- | --- |
| Algoritmo | PPO estilo PureJaxRL (Flax y Optax): actor y crítico MLP 2×128 con tanh; versión recurrente con codificador 128, GRU 128 y cabezas separadas | Implementación de referencia del HANDOFF (§13) |
| Hiperparámetros | 8 192 entornos sincronizados, rollouts de 25 latidos (12 por episodio de 300), 4 épocas, minibatches de 8 192 transiciones (recurrente: 32 grupos de 256 entornos con la secuencia completa), lr 3e-4 con decaimiento lineal, γ 0.99, λ 0.95, clip 0.2, entropía 0.01, coeficiente de valor 0.5, recorte de gradiente 0.5; 30 M latidos (~11 min por agente) | Los valores de Li et al. 2024 (Tabla 2), con lotes más grandes porque el entorno está limitado por latencia (48 000 latidos/s con 8 192 entornos) |
| Truncación | El último paso de cada episodio se trata como truncación: se usa el valor de la observación final y no se enmascara | Los episodios de 300 latidos son un límite de tiempo, no un estado terminal |
| Agentes | oracle (bomba + estado de la RM), oracle_p (la misma observación, pero con presiones en vez de los estadísticos de flujo), rnn (GRU, solo bomba), mlp (solo bomba, sin memoria), fz (presiones, ±1%, recompensa r_A + r_B) y fz4 (igual, con ±1% y ±5%) | §9 y §11. oracle_p aísla el efecto del sensor y es la referencia para H3. En el MDP de Fernández-Zapico la acción 0 siempre obtiene la recompensa máxima, así que la variante de "5 niveles" usa 4 niveles sin el 0 |
| Evaluación | 2 048 escenarios no vistos (semilla 12345), política greedy, todas las políticas parten en la velocidad del clínico (salvo la constante poblacional), métricas con las etiquetas verdaderas y la RM con N = 5; además los 3 protocolos de 1 206 latidos | Misma tubería que las líneas base (`lvadrm.evaluation`) |

## Resultados principales (escenarios no vistos, N = 5)

Todas las políticas parten en la velocidad del clínico (0.97 × la mayor velocidad que abre la válvula al inicio), salvo la constante poblacional. Una primera evaluación con una velocidad inicial común mostró que el 57% de las violaciones del oracle venía del transitorio inicial; por eso se cambió el inicio (decisión del checkpoint 3).

| Política | Episodios con violación | Violaciones / 100 latidos | AVO | BACK | LOW | Gasto medio | Retorno RM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Velocidad constante poblacional | 41% | 6.22 | 69% | 21% | 38% | 5.11 | −56 |
| Velocidad constante del clínico | 30% | 2.80 | 86% | 12% | 38% | 5.26 | −35 |
| Simaan | 45% | 4.38 | 78% | 20% | 37% | 5.47 | −54 |
| fz / fz4 | 22% / 23% | 2.87 / 3.03 | 85% / 83% | 59% / 57% | 52% / 51% | 4.78 / 4.84 | −106 / −104 |
| mlp (sin memoria) | 12% | 0.73 | 88% | 0% | 41% | 4.84 | −8.9 |
| rnn (GRU, solo bomba) | 20% | 0.50 | 97% | 0% | 43% | 4.78 | −8.6 |
| **oracle** (bomba + RM) | **10%** | **0.07** | 68% | 1% | 35% | 5.02 | **−8.5** |
| **oracle_p** (presiones + RM) | **0%** | **0.01** | 47% | 0% | 35% | 5.29 | −8.8 |

Protocolos, violaciones cada 100 latidos (ejercicio / hipertensión / secuencial):

- **oracle_p:** 0 / 0 / 0.
- **oracle:** 0 / 0.8 / 0.
- **mlp y rnn:** 0 / 15.1–15.2 / 0.
- **Clínico:** 0.1 / 15.1 / 5.1.
- **Simaan:** 15.6 / 0 / 7.4.
- **fz y fz4:** 0 / 0 / 0.5, pero con BACK en 76–99% de los latidos.

## Interpretación

- **Punto de decisión (§14): se supera.** El oracle tiene 40 veces menos violaciones que el clínico (0.07 contra 2.80 cada 100 latidos) y 3 veces menos episodios con violación, sin succión. No hace falta simplificar.
- **H1 (estructura de RM contra memoria sola): a favor.** El oracle tiene 7–10 veces menos violaciones que rnn y mlp, y abre la válvula solo lo necesario (68% de los latidos), lo que le deja más gasto: 5.02 contra 4.78–4.84 l/min, y LOW en 35% contra 41–43%. rnn y mlp llegan a un retorno parecido con una estrategia conservadora, que es dejar la válvula abierta casi siempre (88–97%). Esa estrategia es markoviana, por eso la memoria de rnn apenas supera a mlp. La diferencia clara está en hipertensión: ahí no existe velocidad constante que abra la válvula sin regurgitar, y solo los oracles modulan la velocidad para abrirla periódicamente aceptando BACK (0–0.8 contra ~15 violaciones cada 100 latidos). Con una sola semilla todavía falta la incertidumbre estadística (Fase 5).
- **H3 (sensores):** con presiones, el oracle_p reduce las violaciones de 0.07 a 0.01 cada 100 latidos y no tiene ningún episodio con violación. Esa es la brecha de sensor que el agente TDM, con solo señales de bomba, debería cerrar. El oracle_p es la referencia privilegiada para H3; la réplica de Fernández-Zapico es débil con nuestras métricas porque su recompensa no penaliza el reflujo (BACK en 57–59% de los latidos, 99% en hipertensión).

## Pendiente o por decidir

- 5 semillas por agente (Fase 5).
- En entrenamiento, la velocidad inicial sigue siendo aleatoria alrededor de la del paciente base (× 0.9–1.1); calcular la del clínico en cada reset costaría ~80% más de simulación.
