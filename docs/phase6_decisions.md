# Fase 6: robustez y generalización

Complementa los resúmenes en [results/phase6/](../results/phase6/): `summary.md` (A), `summary_eval_wide.md` y `summary_eval_lowemax.md` (B), `summary_eval_noise.md` (C) y `summary_eval_N.md` (D), además de los `aggregate_*.json`, `tdm/*.json` y `hypertension_diagnosis.json`. Se reproduce con `python scripts/collect_data_v2.py` y `python scripts/run_phase6.py --stages A B C D` (~11 h en la 3090).

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

## Problemas y pendientes

1. **Colapso de PPO en condiciones difíciles:** 5 corridas de 66 en la Fase 6, todas en la semilla 2. Posibles remedios: normalizar la recompensa o la ventaja por escalas, un piso de entropía, reiniciar las corridas atascadas (por ejemplo, si AVO se mantiene en 0 durante N latidos tras 5 M latidos), o reportar la tasa de colapso como métrica. Por ahora se reporta aparte.
2. **Generalización fuera de rango:** entrenar los agentes, y no solo el TDM, con escenarios amplios.
3. **Baja contractilidad:** revisar ω_min para corazones muy débiles antes de usarla como test.
