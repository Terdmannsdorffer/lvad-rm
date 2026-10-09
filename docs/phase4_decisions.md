# Fase 4: detectores, inferencia y agentes con creencia

Complementa [results/phase4/summary.md](../results/phase4/summary.md), [results/phase4/tdm/inference.json](../results/phase4/tdm/inference.json) y [results/phase4/detectors/detectors.json](../results/phase4/detectors/detectors.json). Una semilla por agente.

## Decisiones tomadas

| Tema | Decisión | Justificación |
| --- | --- | --- |
| Foco | Detectores e inferencia solo para AVO | Decisión (a) del checkpoint 2. Además, la RM de conteo solo transiciona con AVO: IBU necesita únicamente P(AVO) |
| Datos | 2 048 / 256 / 512 episodios (entrenamiento / validación / prueba) de 300 latidos; mitad con acciones aleatorias y mitad con el oracle estocástico; ~845 000 latidos (fuera de git) | Cobertura amplia (violaciones) y estados cerca de la zona de operación |
| Señal observable | `info["qp_wave"]`: las 64 muestras con ruido de Qp del latido (la misma medición de la que salen los estadísticos de la observación) | Es lo que mide un sistema real |
| Detectores aprendidos | Logístico (13 estadísticos), MLP y CNN 1D (estadísticos + onda); entropía cruzada binaria | Tres niveles de calidad |
| Detectores sintéticos | Independiente (ε 0.1 y 0.2), Markov (ε 0.1 y 0.2, persistencia 0.9), por paciente (ε 0.2 ± 100%) y por margen de apertura (0.5 · exp(−\|V_ao − 0.5\| / 1 ml)). Su salida p = P(AVO \| d) se calibra en el conjunto de entrenamiento | Ruido controlado para H2: un detector calibrado latido a latido que ignora la estructura temporal de sus errores |
| TDM | Codificador 2×128 → GRU 128 → softmax sobre los 5 estados; entrenado con el estado verdadero de la RM sobre episodios completos (30 épocas, Adam 1e-3) | Arquitectura de Li et al. simplificada |
| TDM-det | La misma red, pero solo con la salida del detector + velocidad, HR y acciones, sin las señales de flujo | El TDM completo también ve los estadísticos de flujo, que por sí solos predicen AVO (MLP: 97%). TDM-det recibe la misma información sobre AVO que IBU |
| Creencia inicial | one-hot(u₀) al inicio de cada episodio | El estado inicial de la RM es conocido |
| Agentes con creencia | PPO MLP con la observación de bomba + vector de creencia (Naive: one-hot de û; IBU: creencia exacta; TDM: softmax). Mismos hiperparámetros que la Fase 3 | §10 |

## Resultados

**Detectores (prueba).** Columnas: exactitud, especificidad, AUC, P(error) y P(error \| error en el latido anterior).

- **MLP:** 0.97, 0.96, 0.997, 0.029 y 0.13.
- **CNN:** 0.97, 0.94, 0.996, 0.034 y 0.16.
- **Logístico:** 0.79, 0.54, 0.87, 0.21 y 0.54.

Los errores de los detectores aprendidos están correlacionados en el tiempo (4–5 veces más probables después de otro error), que es la premisa de H2. El logístico da muchos falsos positivos.

**Calidad de la creencia** (log-verosimilitud del estado verdadero, piso ln 0.01; Naive / IBU / TDM-det / TDM):

| Detector | Log-verosimilitud | Creencia en u = 0 con la válvula cerrada |
| --- | --- | --- |
| MLP | −0.53 / −0.31 / −0.19 / −0.18 | 0.10 / 0.14 / 0.08 / 0.09 |
| Logístico | −1.49 / −0.89 / −0.42 / −0.30 | 0.55 / 0.49 / 0.18 / 0.11 |
| Independiente ε 0.2 | −1.61 / −0.92 / −0.29 / −0.23 | 0.25 / 0.43 / 0.13 / 0.08 |
| Markov ε 0.2 | −1.18 / −0.84 / −0.15 / −0.10 | 0.24 / 0.43 / 0.06 / 0.04 |

**Control** (2 048 escenarios no vistos, N = 5, inicio en la velocidad del clínico):

| Agente | Violaciones / 100 latidos (MLP / logístico / Markov 0.2) | AVO | Gasto medio |
| --- | --- | --- | --- |
| oracle | 0.07 | 68% | 5.02 |
| oracle con presiones | 0.01 | 47% | 5.29 |
| rnn (sin RM) | 0.50 | 97% | 4.78 |
| Naive | 0.64 / 0.59 / 0.38 | 87–90% | 4.82–4.90 |
| IBU | 0.62 / 0.56 / 0.35 | 85–89% | 4.82–4.91 |
| TDM-det | — / — / 0.48 | 91% | 4.82 |
| TDM | **0.30 / 0.49 / 0.20** | 70–80% | 4.93–4.98 |

En el protocolo de hipertensión, TDM es el único agente con creencia que baja de ~15 violaciones cada 100 latidos (Markov 5.6, logístico 10.8, MLP 15.0); el oracle queda en 0.8.

## Interpretación (preliminar, una semilla)

- **H2, a nivel de creencia: confirmada.** Naive e IBU sobrestiman la apertura: le asignan 0.24–0.55 de probabilidad a "recién abrió" cuando la válvula seguía cerrada, contra 0.04–0.18 con TDM. La ventaja de TDM-det sobre IBU crece cuando los errores son persistentes (Markov contra independiente con el mismo ε).
- **H2, a nivel de control: matizada.** Con la creencia de Naive o IBU, PPO no empuja la velocidad hacia arriba como predecía H2: aprende a desconfiar de ella y se vuelve conservador (válvula abierta en 85–91% de los latidos, menos gasto), igual que el rnn sin RM. El costo de una mala creencia aparece como conservadurismo, no como más violaciones. TDM, en cambio, se comporta más parecido al oracle (abre menos, entrega más gasto) y tiene el menor número de violaciones entre los agentes sin oráculo: aproximadamente la mitad que Naive e IBU con los detectores MLP y Markov.
- **TDM-det mejora mucho la creencia pero no el control** (0.48 contra 0.35–0.38 de Naive e IBU con Markov). Hace falta ver si es por la semilla (Fase 5).
- **H3: todavía lejos.** El mejor agente con solo señales de bomba (TDM) tiene 0.20–0.30 violaciones cada 100 latidos, contra 0.07 del oracle y 0.01 del oracle con presiones.
- **Los retornos de la RM casi no separan a los agentes** (−7.9 a −9.5). Con r_back = −0.5 por latido y r_viol = −1 por violación, evitar BACK pesa más que evitar violaciones: el oracle abre la válvula aceptando 1% de latidos con BACK (≈ −1.5 por episodio), mientras que las políticas conservadoras no tienen BACK. **Decisión pendiente** sobre los pesos (ver el checkpoint).
