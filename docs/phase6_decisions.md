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
- **Corrección posterior (sección G):** el colapso venía del recorte del valor en PPO. Sin recorte, Naive e IBU con la semilla 2 no colapsan, así que no se le puede atribuir a la inferencia de TDM. Lo que queda de H2 aquí es la ventaja moderada en violaciones y la política menos conservadora.

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
- **El retorno del oracle no es el mejor.** Naive e IBU logran un retorno de la RM algo mayor (−7.0 a −7.5 contra −7.8) con una política más conservadora. Descomposición del retorno por episodio, oracle contra Naive con errores independientes:

  | Término | oracle | Naive |
  | --- | --- | --- |
  | Violaciones | −0.03 | −0.31 |
  | Reflujo | −0.59 | −0.28 |
  | Gasto | −7.02 | −6.34 |
  | Acción | −0.11 | −0.03 |

  - El oracle pierde por seguir peor el gasto objetivo (\|CO − 5\| = 1.17 contra 1.06 l/min) y por el reflujo de sus bajadas de velocidad, no por violaciones.
  - Subir w_perf agrandaría la brecha, así que no es un problema de pesos.
  - La política conservadora también estaba al alcance del oracle. El oracle converge hacia los 8 M latidos a un régimen intermitente (velocidad alta con bajadas para abrir la válvula) y después casi no cambia (KL ≈ 0): es un óptimo local de PPO.
- **Para el paper:** el escenario con detector clínico y errores persistentes da el resultado más limpio para H2 (hipertensión). Hay que declarar que los errores sintéticos no dependen del régimen y que la estructura temporal de los errores clínicos es desconocida.

## G. Colapso de PPO: el recorte de la función de valor

**Diagnóstico.** Las 5 corridas que colapsaron en C y D, todas con la semilla 2, siguen el mismo patrón:

1. En ~2 M latidos la política baja a la velocidad mínima: la válvula abre siempre, hay reflujo en 89% de los latidos y la entropía cae.
2. A la velocidad mínima las acciones de bajar no hacen nada, y la política deriva hacia el extremo opuesto (~13 500 rpm), donde la válvula nunca abre.
3. La pérdida del crítico salta a 3–10 y queda alta durante más de 20 M latidos.

La causa probable es el recorte del valor estilo PPO:

- Recortaba el cambio del valor a ±0.2 por actualización, en unidades absolutas.
- Los valores van de ~−2.5 en el régimen normal a ~−30 en el colapsado, y en todo el entrenamiento hay 146 actualizaciones. Si la política salta de régimen, el crítico tarda casi todo el entrenamiento en alcanzarla, y mientras tanto las ventajas son ruido.
- Andrychowicz et al. (ICLR 2021, sección 3.4) encontraron que este recorte "hurts the performance regardless of the clipping threshold" y recomiendan no usarlo.

**Qué se hizo:**

- `PPOConfig.vf_clip` (por defecto 0.2, el comportamiento anterior) y la opción `--no-vf-clip` de `train_ppo.py`.
- Se reentrenaron sin recorte las 5 corridas colapsadas y 3 semillas del oracle (`run_phase6.py --stages G`, `results/phase6/eval_novfclip.json`).

**Resultados:**

| Corrida | Con recorte | Sin recorte | Otras semillas, con recorte |
| --- | --- | --- | --- |
| Naive, independiente ε 0.3, s2 | 19.7 | 0.16 | 0.14–0.15 |
| IBU, independiente ε 0.3, s2 | 19.7 | 0.14 | 0.15–0.16 |
| IBU, Markov ε 0.3, s2 | 19.7 | 0.18 | 0.08–0.16 |
| oracle N = 3, s2 (violaciones con N = 3) | colapsada | 0.01 | 0.02–0.03 |
| TDM N = 3, s2 (violaciones con N = 3) | colapsada | 0.14 | 0.14–0.15 |

(violaciones cada 100 latidos en los 2 048 escenarios de prueba)

- **Las 5 corridas dejan de colapsar** y quedan al nivel de las semillas que no habían colapsado. La pérdida del crítico se mantiene bajo 0.8 durante todo el entrenamiento.
- **El oracle sin recorte mejora en su propio objetivo** (3 semillas contra las 5 de la Fase 5):

  | Métrica | Con recorte | Sin recorte | p |
  | --- | --- | --- | --- |
  | Retorno de la RM | −7.75 | −7.19 | 0.08 |
  | Reflujo | 2.0% | 0.9% | 0.03 |
  | \|CO − 5\| (l/min) | 1.17 | 1.12 | 0.02 |
  | Violaciones / 100 latidos | 0.009 | 0.039 | 0.15 |

  - Cambia un poco más de violaciones, todavía muy pocas, por menos reflujo y mejor seguimiento del gasto.
  - En los tres protocolos, incluida la hipertensión, tiene 0 violaciones.
  - La semilla 2 sale del régimen intermitente (AVO 74%) y llega a −6.81, mejor que cualquier agente con creencia. Las semillas 0 y 1 siguen en él (−7.38).
  - El óptimo local se atenúa, pero no desaparece.

**Implicación.** Todas las corridas anteriores usan el recorte. Para comparaciones homogéneas hay que reentrenar sin recorte, como mínimo el experimento principal y la sección F. Mientras tanto, los colapsos de C y D deben leerse como un artefacto de la implementación de PPO y no de la inferencia.

## H. Reentrenamiento completo sin recorte del valor

**Qué se hizo:**

- Se reentrenaron sin recorte del valor todos los agentes PPO de las Fases 5 y 6: experimento principal, F, C y D. Luego se repitieron las evaluaciones, incluida la generalización de B.
- Script: `scripts/run_noclip.py`, ~33 h en la 3090. Resultados en `results/phase6/noclip/`, con los mismos nombres de archivo que en las Fases 5 y 6.
- Las 8 corridas de la sección G tienen la misma configuración y se copiaron en vez de reentrenarse.
- **Sin cambios:** detectores y modelos TDM. El TDM v2 se entrenó en parte con datos de la política TDM de la Fase 5, con recorte. Se mantuvo, porque es solo un conjunto de datos para el modelo de inferencia.
- **Copiadas de las evaluaciones con recorte:** las líneas base clásicas y las reglas de modulación, que son deterministas y usan el mismo conjunto de prueba.

**Experimento principal** (5 semillas; violaciones cada 100 latidos; con recorte → sin recorte):

| Agente | Violaciones | Retorno RM | Hipertensión |
| --- | --- | --- | --- |
| oracle | 0.01 → 0.03 | −7.8 → **−7.2** | 0 → 0 |
| oracle con presiones | 0.01 → 0.01 | −10.4 → −9.8 | 0 → 0 |
| TDM v2 | 0.17 → **0.13** | −7.9 → −7.6 | 6.3 → 5.4 |
| TDM (Fase 4) | 0.18 → 0.15 | −7.8 → −7.7 | 14.5 → 14.8 |
| IBU | 0.22 → 0.19 | −7.6 → −7.5 | 14.6 → 15.0 |
| Naive | 0.23 → 0.26 | −8.0 → −8.0 | 14.2 → 14.2 |
| PPO recurrente | 0.20 → 0.35 | −8.1 → −8.2 | 7.0 → 3.5 |

- **El oracle vuelve a tener el mejor retorno entre los agentes de RL.** Abre la válvula en 71% de los latidos (antes 62%), con 1.3% de reflujo y el gasto en el objetivo (5.04 l/min).
- **H1 se fortalece:** oracle contra PPO recurrente, −0.33 violaciones cada 100 latidos (p = 0.016). En parte es porque el recurrente empeora: tiene más violaciones en sus 5 semillas. El recorte del valor parece ayudar a un crítico recurrente de alta varianza, así que esto no es una mejora general.
- **H2 se fortalece:**
  - TDM v2 contra IBU: −0.053 (p = 0.01); con recorte no era significativo.
  - TDM contra Naive: −0.11 (p = 0.002).
- **H3 no cambia:** TDM v2 contra el oracle, +0.11 (p < 0.001).

**Detector clínico (F):**

| Detector | Naive | IBU | TDM | Hipertensión (Naive / IBU / TDM) |
| --- | --- | --- | --- | --- |
| Persistente | 0.13 | 0.13 | **0.09** | 13.2 / 14.7 / **0.03** |
| Independiente | 0.13 | 0.12 | 0.10 | **0.95** / 1.23 / 4.05 |

- **Con errores persistentes TDM ahora gana también en escenarios aleatorios:** contra IBU p = 0.015, contra Naive p = 0.004. Con recorte era un empate. En hipertensión sigue siendo el único que resiste.
- **Con errores independientes TDM ya no es peor en escenarios aleatorios:** contra IBU p = 0.17, contra Naive p = 0.045. En hipertensión Naive e IBU siguen siendo mejores, lo que coincide con Li et al. (2024).
- Los retornos de oracle, TDM, Naive e IBU quedan entre −7.0 y −7.3.
- Las reglas de modulación con el mismo detector quedan igual que en F (0.20 y 0.29).

**Barrido de ruido (C, 3 semillas):**

| Detector | Naive | IBU | TDM | p (TDM − IBU) |
| --- | --- | --- | --- | --- |
| Independiente ε 0.1 | 0.17 | 0.19 | **0.09** | 0.004 |
| Independiente ε 0.3 | 0.16 | 0.13 | 0.13 | 0.72 |
| Markov ε 0.1 | 0.13 | 0.14 | 0.16 | 0.56 |
| Markov ε 0.3 | 0.18 | 0.17 | **0.08** | 0.002 |

- Ninguna corrida colapsa (0 de 36).
- TDM gana con claridad en 2 detectores y empata en los otros 2. Tiene más varianza entre semillas (Markov ε 0.1: 0.10–0.22).
- En los 4 detectores TDM entrega más gasto (4.90–4.96 contra ~4.80 l/min) y fuerza menos aperturas.
- Sin los colapsos, la ventaja de TDM en violaciones en el barrido sintético es moderada e inconsistente.

**Horizonte N (D, 3 semillas, cada uno con su N):**

| Agente | N = 3 | N = 5 | N = 10 |
| --- | --- | --- | --- |
| oracle | 0.011–0.015 | 0.013–0.053 | 0.003–0.005 |
| TDM | 0.11–0.14 | 0.11–0.13 | 0.10–0.13 |

- Ninguna corrida colapsa (con recorte colapsaron 2 con N = 3).
- La brecha de ~0.1 violaciones cada 100 latidos entre TDM y el oracle se mantiene para todo N.

**Generalización (B):**

- Los agentes con creencia siguen fallando en escenarios amplios (TDM v2: 6.7; Naive, IBU y TDM: 8.5–9.3).
- **El oracle sin recorte generaliza peor que con recorte** (0.80 contra 0.12). Dos de sus semillas tienen 1.1 y 2.1, las otras 0.04–0.39. Su política, menos conservadora, gana dentro del rango de entrenamiento y deja menos margen fuera de él.
- Baja contractilidad: sin cambios (sigue siendo infactible con ω_min).

**Conclusión para el informe:**

1. Los resultados principales se reportan sin recorte (`results/phase6/noclip/`).
2. La evidencia fuerte de H2 es la hipertensión con errores persistentes (0.03 contra 13–15), junto con la ventaja en escenarios aleatorios con detector clínico persistente y con el detector MLP (TDM v2).
3. Con errores independientes la ventaja de TDM desaparece o se invierte en hipertensión, como predice la teoría.

## Problemas y pendientes

1. **Colapso de PPO en condiciones difíciles:** 5 corridas de 66 en las secciones A–D, todas en la semilla 2; en la sección F, ninguna de 30. Resuelto quitando el recorte del valor (sección G); todo se reentrenó sin recorte (sección H).
2. **Generalización fuera de rango:** entrenar los agentes, y no solo el TDM, con escenarios amplios.
3. **Baja contractilidad:** revisar ω_min para corazones muy débiles antes de usarla como test.
4. **El oracle no era una cota superior del retorno:** con recorte, Naive e IBU lo superaban (sección F). La brecha venía del seguimiento del gasto, no de los pesos de la RM. Sin recorte el oracle tiene el mejor retorno del experimento principal (−7.2), aunque en F empata con Naive e IBU (secciones G y H).
5. **Estructura temporal de los errores clínicos:** H2 depende de ella. Si hay registros latido a latido de un detector clínico (por ejemplo, del grupo de Clifford y Hayward), estimar ρ.
6. **Generalización del oracle sin recorte:** dentro del rango de entrenamiento mejora, fuera de él empeora (0.80 contra 0.12 en escenarios amplios). Entrenar con escenarios amplios (punto 2) es ahora más importante.
