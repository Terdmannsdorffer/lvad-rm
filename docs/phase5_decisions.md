# Fase 5: experimento principal (5 semillas)

Complementa [results/phase5/summary.md](../results/phase5/summary.md), [results/phase5/aggregate.json](../results/phase5/aggregate.json) y [results/phase5/hypertension_diagnosis.json](../results/phase5/hypertension_diagnosis.json).

## Configuración

- **Agentes:** oracle (bomba + estado de la RM), oracle_p (presiones + estado de la RM), rnn (GRU, sin RM), y Naive, IBU y TDM con el detector MLP aprendido en la Fase 4. 5 semillas cada uno; 30 M latidos por corrida, ~11 min en la 3090.
- **Recompensa:** RM con N = 5, r_viol −1, r_suc −2, **r_back −0.1** (decisión del checkpoint 4; antes −0.5), w_perf 0.1, w_act 0.01, gasto objetivo 5 l/min.
- **Evaluación:** 2 048 escenarios no vistos (semilla 12345) y 3 protocolos de 1 206 latidos; política greedy; inicio en la velocidad del clínico. Se reportan media ± error estándar sobre semillas, IC bootstrap del 95% en `aggregate.json` y t de Welch para las comparaciones.

## Resultados (escenarios no vistos)

| Política | Violaciones / 100 latidos | Episodios con violación | AVO | BACK | Gasto (l/min) |
| --- | --- | --- | --- | --- | --- |
| Velocidad constante poblacional | 6.22 | 41% | 69% | 21% | 5.11 |
| Velocidad constante del clínico | 2.80 | 30% | 86% | 12% | 5.26 |
| Simaan | 4.38 | 45% | 78% | 20% | 5.47 |
| oracle_p | 0.01 ± 0.00 | 0% | 56 ± 3% | 8.6 ± 1.8% | 5.19 ± 0.03 |
| oracle | 0.01 ± 0.00 | 1% | 62 ± 2% | 2.0% | 5.13 ± 0.01 |
| TDM (MLP) | 0.18 ± 0.03 | 15 ± 1% | 62 ± 5% | 1.3% | 5.07 ± 0.03 |
| IBU (MLP) | 0.22 ± 0.01 | 15 ± 1% | 75 ± 3% | 1.4% | 4.99 ± 0.01 |
| Naive (MLP) | 0.23 ± 0.01 | 19 ± 1% | 67 ± 1% | 1.9% | 5.01 ± 0.01 |
| rnn | 0.20 ± 0.06 | 18 ± 5% | 90 ± 4% | 2.7% | 4.93 ± 0.05 |

Protocolos, violaciones cada 100 latidos (ejercicio / hipertensión / secuencial):

- **Oracles:** 0 / 0 / 0.
- **TDM, IBU y Naive:** 0 / 14.2–14.6 / 0–0.15.
- **rnn:** 0 / 7.0 ± 2.3 / 0.
- **Clínico:** 0.1 / 15.1 / 5.1.

## Comparaciones (Welch, n = 5 contra 5)

| Comparación | Δ violaciones / 100 | Δ gasto | Δ AVO |
| --- | --- | --- | --- |
| H1: oracle − rnn | −0.19 (p = 0.027) | +0.20 (p = 0.010) | −28 pp (p < 0.001) |
| H2: TDM − IBU | −0.03 (p = 0.31) | +0.08 (p = 0.087) | −14 pp (p = 0.063) |
| H2: TDM − Naive | −0.05 (p = 0.16) | +0.07 (p = 0.13) | −6 pp (p = 0.35) |
| TDM − rnn | −0.02 (p = 0.76) | +0.14 (p = 0.041) | −28 pp (p = 0.003) |
| H3: TDM − oracle | +0.17 (p = 0.004) | −0.06 (p = 0.16) | 0 pp (p = 0.97) |
| H3: TDM − oracle_p | +0.18 (p = 0.004) | −0.11 (p = 0.032) | +6 pp (p = 0.38) |
| oracle_p − oracle | −0.003 (p = 0.20) | +0.05 (p = 0.12) | −6 pp (p = 0.17) |

## Interpretación

- **H1: sí.** Conocer el estado de la RM permite cumplir la restricción con menos aperturas: el oracle abre en 62% de los latidos contra 90% del rnn, entrega 0.2 l/min más y tiene menos violaciones, todo con diferencias significativas.
- **Con r_back −0.1, el oracle de bomba alcanza al de presiones** (0.01 violaciones cada 100 latidos en ambos; en la Fase 3, con −0.5, eran 0.07 contra 0.01). La brecha de sensor que se veía en la Fase 3 era en realidad un efecto de los pesos de la recompensa. Conociendo el estado de la RM, las señales de bomba bastan.
- **H2 a nivel de creencia: sí** (Fase 4). A nivel de control en escenarios aleatorios, TDM va en la dirección esperada respecto de IBU y Naive (menos violaciones, más gasto, menos aperturas), pero las diferencias no son significativas con 5 semillas. El costo de una creencia peor se paga sobre todo con conservadurismo (más aperturas, menos gasto), no con violaciones.
- **H2 en hipertensión: el mecanismo aparece con claridad.** En el protocolo de hipertensión, el detector MLP da falsos positivos en ~1 de cada 3 latidos cerrados (4% en general): son errores dependientes del estado fisiológico y persistentes. La creencia de TDM e IBU pone ~0.35 de probabilidad en "recién abrió" cuando la válvula está cerrada y solo ~0.4 en el estado de riesgo cuando el agente está en él. El agente no baja la velocidad lo suficiente (~9 025 rpm, contra 8 811 del oracle) y la válvula no vuelve a abrir (AVO ≈ 0; ~14.5 violaciones cada 100 latidos). En este régimen TDM tampoco se salva, porque los datos con que se entrenó no cubren bien esta falla del detector.
- **H3: no se cumple con señales de bomba solas.** TDM queda significativamente por encima de los oracles en violaciones (0.18 contra 0.01). La brecha no es del sensor (los oracles son iguales) sino de la inferencia del estado de la RM.

## Implicancias y siguientes pasos (Fase 6)

1. **Robustez al ruido** con los detectores sintéticos (independientes y correlacionados, barrido de ε): el experimento de H2 que pide el HANDOFF.
2. **Generalización:** escenarios amplios (`WIDE_SCENARIOS`) y menor contractilidad.
3. **Mejorar la inferencia en regímenes de falla del detector.** Por ejemplo, entrenar el TDM con datos que incluyan hipertensión sostenida, o con datos recolectados por la política entrenada (on-policy).
4. **Ablaciones:** N ∈ {3, 10}, espacio de acciones, ruido de medición.
