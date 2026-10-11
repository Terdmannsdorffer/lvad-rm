# Fase 6: robustez y generalización

Complementa los resúmenes en [results/phase6/](../results/phase6/): `summary.md` (A), `summary_eval_wide.md` y `summary_eval_lowemax.md` (B), `summary_eval_noise.md` (C), `summary_eval_N.md` (D) y `summary_eval_clinical.md` (F), además de los `aggregate_*.json`, `tdm/*.json` y `hypertension_diagnosis.json`. Se reproduce con `python scripts/collect_data_v2.py` y `python scripts/run_phase6.py --stages A B C D` (~11 h en la 3090); la sección F es `--stages E` del script (~7 h) y la sección E es `scripts/eval_modulation.py`.

## A. TDM v2: inferencia entrenada con datos amplios y de la propia política

**Por qué.** En la Fase 5, los agentes con creencia fallaban en hipertensión. El protocolo usa Rs × 1.8, fuera del rango de entrenamiento (× 0.44–1.56). En ese régimen el detector MLP da falsos positivos en 31–44% de los latidos cerrados.

**Qué se hizo.** El detector se mantuvo fijo y el TDM se reentrenó con los datos de la Fase 4 más 2 048 episodios nuevos: escenarios amplios con política aleatoria y oracle, y datos de la política TDM de la Fase 5 en escenarios por defecto y amplios. Luego se entrenaron agentes TDM con esta inferencia (5 semillas).

**Creencia** (log-verosimilitud; Naive / IBU / TDM de la Fase 4 / TDM v2):

| Conjunto de prueba | Log-verosimilitud |
| --- | --- |
| Escenarios por defecto | −0.53 / −0.31 / −0.18 / −0.15 |
| Amplios, con oracle | −1.43 / −1.08 / −1.07 / **−0.08** |
| De la propia política, amplios | −1.69 / −1.32 / −1.35 / **−0.40** |

En los escenarios amplios con oracle, la creencia en "recién abrió" con la válvula cerrada baja de 0.48 a 0.04.

**Control** (5 semillas):

- **Escenarios por defecto:** TDM v2 queda igual que el TDM anterior (0.17 ± 0.02 contra 0.18 ± 0.03 violaciones cada 100 latidos).
- **Hipertensión:** 6.3 ± 1.3 contra 14.5 de TDM, IBU y Naive (el oracle tiene 0).
- **Diagnóstico de hipertensión** (semilla 0):
  - la creencia en "recién abrió" con la válvula cerrada baja de 0.35 a 0.12;
  - la creencia en el estado de riesgo cuando el agente está en él sube de 0.37 a 0.66;
  - la velocidad baja de 9 026 a 8 833 rpm (oracle: 8 811);
  - la válvula vuelve a abrir en 25% de los latidos (oracle: 37%).

**Conclusión.** TDM puede aprender a corregir errores del detector que dependen del régimen fisiológico si sus datos cubren ese régimen. Naive e IBU no pueden, por construcción.

## B. Generalización sin reentrenar

**Escenarios amplios** (Rs hasta 2.5 veces el del paciente; entrenamiento: 1.56):

| Política | Violaciones / 100 latidos |
| --- | --- |
| oracle con presiones | 0.01 |
| oracle | 0.12 ± 0.11 |
| Velocidad constante del clínico | 2.87 |
| rnn | 6.1 |
| **TDM v2** | **6.8** |
| TDM | 8.6 |
| Naive | 8.7 |
| IBU | 9.0 |

Con Rs alto, abrir la válvula sin reflujo es imposible, y los agentes que infieren el estado de la RM no generalizan fuera de rango. El clínico, que ajusta la velocidad al comienzo de cada episodio, les gana. El oracle sí generaliza: el límite está en la inferencia, no en la política.

**Baja contractilidad** (Emax × 0.3–0.5; entrenamiento: × 0.5–1.5): todas las políticas fallan, incluido el oracle con 6.6 violaciones cada 100 latidos. Con un corazón tan débil, la válvula aórtica deja de abrir bajo la velocidad mínima permitida (6 000 rpm; por ejemplo, con Emax × 0.5 se cierra en 7 115 rpm), así que buena parte del escenario es infactible con las cotas de acción actuales. No es un buen test de generalización sin bajar ω_min.

## C. Robustez al ruido del detector (H2; 3 semillas)

Violaciones cada 100 latidos (media de las semillas; entre paréntesis, las semillas que colapsaron):

| Detector | Naive | IBU | TDM |
| --- | --- | --- | --- |
| Independiente ε 0.1 | 0.18 | 0.17 | **0.12** |
| Independiente ε 0.3 | 0.15 (+1 colapsada) | 0.16 (+1 colapsada) | **0.10** |
| Markov ε 0.1, ρ 0.9 | 0.15 | 0.16 | **0.08** |
| Markov ε 0.3, ρ 0.9 | 0.18 | 0.12 (+1 colapsada) | 0.12 |

- **TDM tiene la menor cantidad de violaciones en 3 de los 4 detectores y nunca colapsa** (0 de 12 corridas). También entrega más gasto (4.94–5.04 contra 4.78–4.93 l/min) y abre la válvula menos (68–80% contra 82–93%): se acerca al comportamiento del oracle.
- **Naive e IBU colapsan en 3 de 12 corridas con ε 0.3**, todas en la semilla 2: es una política degenerada que no abre nunca la válvula y sube la velocidad (19.7 violaciones cada 100 latidos, gasto ~9 l/min). Cuando no colapsan, compensan la mala creencia volviéndose conservadoras.
- **H2: apoyada.** Con ruido, la inferencia temporal (TDM) es más robusta que Naive e IBU, sobre todo porque evita el colapso del aprendizaje y porque permite una política menos conservadora. La ventaja en violaciones entre corridas que no colapsan es moderada.

## D. Ablación del horizonte N (3 semillas; cada agente evaluado con su propio N)

| Agente | N = 3 | N = 5 | N = 10 |
| --- | --- | --- | --- |
| oracle | 0.02–0.03 (+1 colapsada) | 0.01 | 0.004–0.007 |
| TDM | 0.14–0.15 (+1 colapsada) | 0.15–0.20 | 0.12–0.18 |

- **La brecha entre TDM y el oracle (~0.1–0.15 violaciones cada 100 latidos) se mantiene para todo N.**
- **Con N = 3 la semilla 2 colapsó tanto para el oracle como para TDM**, con la misma política degenerada. N = 3 es más difícil de aprender porque las violaciones llegan antes y son más frecuentes.
- **Con N = 10 los agentes abren menos** (oracle: 51% de los latidos contra 61% con N = 5).

## E. Líneas base de modulación intermitente de velocidad

**Por qué.** La revisión del estado del arte mostró que el competidor natural no basado en RL es la modulación intermitente de velocidad: lo que hacen las bombas comerciales para favorecer la apertura sin conocer el estado de la válvula, y los controladores de investigación que sí lo detectan.

**Qué se implementó** (`lvadrm/baselines/modulation.py`, `scripts/eval_modulation.py`):

- **Lazo abierto:** la velocidad baja una fracción `drop` durante un latido cada `period` latidos.
- **Regla con detector:** tras `trigger` latidos sin apertura detectada, la velocidad baja `step` por latido hasta detectarla y luego vuelve a la base; la base se adapta. Se probó con el detector MLP y, como cota superior, con la etiqueta verdadera.
- **Ajuste:** grillas de 32 y 36 configuraciones, evaluadas todas sobre los mismos 128 escenarios de validación (semilla 777, distinta de la de prueba). Se eligió la de mayor retorno de la RM:
  - lazo abierto: base sin cambio, caída de 10%, período 5;
  - regla: base sin cambio, trigger 2, paso 5%, adaptación 1%.
- **Evaluación:** idéntica a la de los agentes.

| Política | Violaciones / 100 latidos | Episodios con violación | AVO | BACK | Gasto | Retorno RM | Hipertensión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Velocidad constante del clínico | 2.80 | 30% | 86% | 11.8% | 5.26 | −20.3 | 15.1 |
| Modulación en lazo abierto | 0.96 | 12% | 89% | 19.2% | 5.17 | −17.4 | **0.0** |
| Regla con detector MLP | 0.65 | 27% | 95% | 5.9% | 5.26 | −12.2 | 14.3 |
| Regla con etiqueta verdadera | 0.01 | 0% | 93% | 4.0% | 5.32 | −9.9 | 0.0 |
| **TDM v2** (RL, solo bomba) | **0.17 ± 0.02** | 16% | 69% | 2.1% | 5.02 | **−7.9** | 6.3 ± 1.3 |
| oracle (RL) | 0.01 | 1% | 62% | 2.0% | 5.13 | −7.8 | 0.0 |

**Interpretación:**

- Con el mismo detector, **TDM tiene ~4 veces menos violaciones que la regla** (0.17 contra 0.65), abre la válvula en 69% de los latidos contra 95%, y regurgita menos (2.1% contra 5.9%). La regla toma cada detección al pie de la letra y, en hipertensión, cae en la misma trampa de los falsos positivos (14.3 violaciones cada 100 latidos). Esto refuerza H2 contra un competidor realista.
- **La modulación en lazo abierto no depende del detector:** no falla en hipertensión, pero lo paga con más violaciones en general (0.96) y mucho reflujo (19% de los latidos).
- **Con información perfecta, la regla simple iguala al oracle de RL en violaciones (0.01).** Pero abre la válvula en casi todos los latidos (93% contra 62%) y regurgita el doble. El aporte del RL con estado de la RM está en el compromiso: menos aperturas forzadas, menos reflujo y gasto cerca del objetivo (retorno −7.8 contra −9.9). El aporte de la inferencia temporal está en la robustez frente al detector (TDM contra la regla con el detector MLP).
- **Para el paper:** estas líneas base son deterministas (n = 1) y fueron ajustadas en validación; los agentes de RL tienen 5 semillas. Conviene declararlo así.

## F. Detector de nivel clínico (5 semillas)

**Por qué.** El detector MLP es optimista frente a la detección clínica de la apertura aórtica (AUC 0.997 contra 0.95). Clifford et al. (2019) reportan, en 18 pacientes con HVAD, sensibilidad 86% y especificidad 93% por latido (solo se leyó el resumen). Este es el escenario principal propuesto en la revisión bibliográfica.

**Qué se hizo:**

- **Detectores sintéticos con esas tasas:** falsos negativos 0.14 y falsos positivos 0.07 (`SyntheticDetector` con `eps_fp`), con dos estructuras de error:
  - independientes (`iid_fn0.14_fp0.07`);
  - persistentes, con un proceso de Markov de ρ = 0.9 (`markov_fn0.14_fp0.07_r0.9`).
- **Supuesto:** no sabemos cómo se distribuyen en el tiempo los errores del detector clínico. Los sintéticos no dependen del régimen fisiológico; los del MLP sí.
- **Inferencia:** un TDM v2 por detector, con los mismos datos de la sección A.
- **Agentes:** Naive, IBU y TDM, 5 semillas por detector.
- **Línea base:** la regla de modulación con cada detector en el lazo, con la misma grilla y validación que en E.
- **Evaluación:** el mismo conjunto de prueba; las referencias se copiaron de `eval.json`.

**Calidad de la creencia** (log-verosimilitud del estado verdadero de la RM sobre datos on-policy; escenarios por defecto / amplios):

| Detector | Naive | IBU | TDM v2 |
| --- | --- | --- | --- |
| Persistente | −0.47 / −1.14 | −0.50 / −1.10 | **−0.02 / −0.07** |
| Independiente | −0.76 / −1.93 | −0.62 / −1.29 | **−0.08 / −0.29** |

**Control** (media ± error estándar sobre 5 semillas; las reglas son deterministas):

| Política | Violaciones / 100 latidos | Episodios con violación | AVO | BACK | Gasto | Retorno RM | Hipertensión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | 0.01 | 1% | 62% | 2.0% | 5.13 | −7.8 | 0.00 |
| TDM v2, detector MLP | 0.17 ± 0.02 | 16% | 69% | 2.1% | 5.02 | −7.9 | 6.30 ± 1.29 |
| Persistente: Naive | 0.13 ± 0.01 | 10% | 86% | 1.7% | 4.84 | −7.4 | 10.85 ± 0.70 |
| Persistente: IBU | 0.10 ± 0.02 | 9% | 83% | 2.1% | 4.86 | −7.5 | 14.53 ± 0.23 |
| Persistente: TDM | 0.10 ± 0.01 | 16% | 64% | 1.9% | 5.06 | −7.8 | **0.10 ± 0.03** |
| Persistente: regla | 0.20 | 19% | 92% | 10.4% | 5.23 | −12.3 | 0.17 |
| Independiente: Naive | 0.10 ± 0.00 | 15% | 89% | 0.9% | 4.87 | −7.0 | 1.61 ± 0.42 |
| Independiente: IBU | 0.11 ± 0.01 | 14% | 90% | 0.9% | 4.86 | −7.0 | 3.15 ± 1.29 |
| Independiente: TDM | 0.14 ± 0.01 | 18% | 71% | 1.5% | 5.02 | −7.7 | 4.34 ± 0.71 |
| Independiente: regla | 0.29 | 48% | 93% | 4.9% | 5.30 | −11.0 | 0.00 |

**Interpretación:**

- **Con error clínico ningún agente colapsa** (0 de 30). Todos tienen pocas violaciones en escenarios aleatorios (0.10–0.14), menos que TDM v2 con el detector MLP (0.17), pese a que el MLP tiene mayor AUC. Los errores sintéticos están repartidos sin relación con la fisiología; los del MLP se concentran en ciertos regímenes (falsos positivos en hipertensión). Para el control importa más dónde caen los errores que su tasa promedio.
- **Errores persistentes en escenarios aleatorios:** TDM empata en violaciones con IBU (0.10 contra 0.10, p = 0.72) y con Naive (0.13, p = 0.12). Su política, en cambio, se parece a la del oracle: abre la válvula en 64% de los latidos contra 83–86%, y entrega 5.06 l/min contra 4.84–4.86 (p < 10⁻⁶). Naive e IBU compensan su mala creencia con conservadurismo: más aperturas forzadas y gasto bajo el objetivo.
- **Errores persistentes en hipertensión:** TDM tiene 0.10 violaciones cada 100 latidos, contra 10.9 de Naive y 14.5 de IBU. Es el mecanismo de H2: una racha de falsos positivos convence a Naive e IBU de que la válvula abrió. IBU supone errores independientes y es el más sobreconfiado. TDM también ve el flujo de la bomba y no cae.
- **Con errores independientes TDM no gana.** Tiene algo más de violaciones que Naive e IBU (0.14 contra 0.10–0.11, p ≈ 0.03–0.04) y más en hipertensión (4.3 contra 1.6–3.2), aunque mantiene el gasto en el objetivo. Esto concuerda con Li et al. (2024): la ventaja de TDM aparece con errores correlacionados. Su creencia sigue siendo mejor en log-verosimilitud, así que la diferencia probablemente viene de una política menos conservadora y no de la inferencia (hipótesis, no verificada).
- **Contra la regla con el mismo detector**, los agentes tienen 1.5–3 veces menos violaciones (0.10–0.14 contra 0.20–0.29), 3–6 veces menos reflujo y mejor retorno (−7.0 a −7.8 contra −11.0 a −12.3). En hipertensión la regla rinde bien (0.17 y 0.00): abre la válvula en 92–93% de los latidos, y ese margen la protege. Con errores independientes, la regla supera en hipertensión a los tres agentes.
- **El retorno del oracle no es el mejor.** Naive e IBU logran un retorno de la RM algo mayor (−7.0 a −7.5 contra −7.8) con una política más conservadora. Con los pesos actuales, la recompensa casi no distingue entre abrir seguido con gasto bajo el objetivo y abrir menos con gasto sobre el objetivo. La política del oracle es un buen óptimo local, pero no el global. Las diferencias son de 0.3–0.8 por episodio de 300 latidos.
- **Para el paper:** el escenario con detector clínico y errores persistentes da el resultado más limpio para H2 (hipertensión). Hay que declarar que los errores sintéticos no dependen del régimen y que la estructura temporal de los errores clínicos es desconocida.

## Problemas y pendientes

1. **Colapso de PPO en condiciones difíciles:** 5 corridas de 66 en las secciones A–D, todas en la semilla 2. En la sección F no colapsó ninguna (0 de 30). Posibles remedios: normalizar la recompensa o la ventaja por escalas, un piso de entropía, reiniciar las corridas atascadas (por ejemplo, si AVO se mantiene en 0 durante N latidos tras 5 M latidos), o reportar la tasa de colapso como métrica. Por ahora se reporta aparte.
2. **Generalización fuera de rango:** entrenar los agentes, y no solo el TDM, con escenarios amplios.
3. **Baja contractilidad:** revisar ω_min para corazones muy débiles antes de usarla como test.
4. **Pesos de la RM:** con detector clínico, Naive e IBU superan al oracle en retorno (sección F). Antes del paper conviene revisar w_perf o el objetivo de gasto, o entrenar el oracle con más latidos, para que el mejor retorno corresponda a la política clínicamente preferida.
5. **Estructura temporal de los errores clínicos:** H2 depende de ella. Si hay registros latido a latido de un detector clínico (por ejemplo, del grupo de Clifford y Hayward), estimar ρ.
