# Fase 6.5: decisiones y resultados

Plan e hipótesis principales en [phase6_5_plan.md](phase6_5_plan.md); se fijaron antes de entrenar los agentes. El código está en `lvadrm/presets.py` (entorno v2), `scripts/phase6_5.py` (calibraciones, datos e inferencia), `scripts/run_phase6_5.py` (pipeline) y `scripts/eval_phase6_5.py` (evaluación clínica).

## Desviaciones del plan

Las hipótesis principales P1–P3 y la métrica principal (tiempo con la válvula cerrada más de 10 s) no cambiaron. Cambiaron cuatro aspectos del entorno o del diseño, todos antes de entrenar los agentes de la comparación:

1. **Escenarios de entrenamiento moderados** (Rs ≤ 1.6 x, volemia ≥ 0.80 x) en vez de amplios: con hipertensión severa no existe un punto de operación aceptable (sección B1).
2. **r_back = −0.3** en vez de −0.1: con N = 3 el peso anterior producía una deriva del entrenamiento hacia baja velocidad (sección B1).
3. **La succión sale de las afirmaciones:** en el modelo solo aparece muy por encima del cierre aórtico (sección B1).
4. **Escala de ruido 4** (no 2) y **detector por latido MLP** (no CNN): la calibración inicial era optimista y la CNN resultó peor (secciones B1 y B2).

## B0. Decisiones tomadas sin reunión con el profesor (supuestos)

Se aplicaron las recomendaciones del plan. Todas son opciones nuevas: el entorno de las Fases 3–6 (v1) no cambia y sus resultados se reproducen bit a bit.

| Decisión | Elegido | Revisar con el profesor |
| --- | --- | --- |
| D1. Restricción | N = 3, que garantiza la razón clínica de apertura 1:3 | Sí |
| D2. Desempeño | Objetivo tipo Starling (abajo) | Sí |
| D3. Observación | Flujo estimado, calibrado a la detección clínica | — |
| D4. Escenarios | Escenarios amplios (Rs hasta 2.5 x) y volemia 0.75–1.10 x | — |
| D5. Semillas | 10 en las comparaciones principales y 5 en las secundarias | — |

## B1. Entorno v2: calibración

**Objetivo de gasto tipo Starling:** co_obj = clip(5.0 + 0.3 (EDP − 12.5), 3.5, 8.0) l/min.

- EDP de referencia: 12.5 mmHg, la del paciente base en reposo a la velocidad del clínico (8 728 rpm, gasto 5.26 l/min).
- Pendiente: 0.3 l/min por mmHg, fijada a mano como supuesto de sensibilidad moderada a la precarga. El criterio automático (la menor pendiente que suba el gasto de equilibrio ≥ 15% en ejercicio) se cumplía trivialmente con 0.1: en ejercicio intenso el corazón nativo ya entrega 6.85 l/min a la velocidad mínima, y 0.1 deja el objetivo casi fijo.
- Equilibrios, donde el gasto iguala al objetivo:

  | Condición | Gasto (l/min) | Velocidad (rpm) | Válvula |
  | --- | --- | --- | --- |
  | Reposo | 5.13 | 8 500 | abierta |
  | Ejercicio liviano | 5.48 | 7 250 | abierta |
  | Ejercicio intenso | 6.85 | 6 000 (cota) | abierta |
  | Hipertensión | 4.11 | 10 250 | cerrada |
  | Hipovolemia (0.85 x) | 4.32 | 8 000 | abierta |

- En hipertensión el equilibrio de gasto cierra la válvula: la recompensa tiene un conflicto real entre gasto y apertura, que es donde la RM importa.

**Succión:**

- Con los escenarios amplios, la succión es alcanzable a la velocidad máxima (16 000 rpm) en 80–89% de los escenarios.
- A velocidades moderadas no aparece nunca: 0% hasta 12 000 rpm, y 15–35% recién a 14 000 rpm.
- La válvula aórtica ya está cerrada a 10 000 rpm en 97% de los escenarios.
- **Una política que respeta la apertura nunca se acerca a la succión.** Es estructural del modelo de Simaan sin ventrículo derecho (Reesink et al. 2007 muestran que la persistencia del colapso depende del ventrículo derecho y de la resistencia pulmonar).
- Se fijó la volemia mínima en 0.75 x para tener más variabilidad de precarga, y **la succión sale de las afirmaciones principales** (plan B del plan). Se sigue midiendo: succiones por hora y familia de hipovolemia.

**Observación de flujo estimado:**

- Errores nominales:
  - ganancia por paciente con desviación de 6% (Karantonis et al. 2007);
  - sesgo de 0.3 l/min (estimador del HVAD);
  - ruido de forma de onda de ~0.7 l/min tras un promedio móvil de 4 muestras.
- Todos se escalan por `flow_noise_scale`. Mejor detector aprendido (CNN sobre la forma de onda y los rasgos, datos de política aleatoria):

  | Escala | AUC | Sensibilidad | Especificidad | P(error \| error anterior) |
  | --- | --- | --- | --- | --- |
  | 1.0 (errores publicados) | 0.969 | 0.932 | 0.846 | 0.57 |
  | 2.0 | 0.953 | 0.885 | 0.854 | 0.59 |
  | Clínica (Clifford et al. 2019) | 0.95 | 0.86 | 0.93 | — |

- **Esta primera pasada quedó superada.** Usó la distribución amplia, 256 episodios y 8 épocas. Con la distribución final y el presupuesto completo del pipeline (512 episodios, 12 épocas), el detector con escala 2 llegó a AUC 0.99.
- **Recalibración** con la distribución final (CNN, 512 episodios de política aleatoria, 12 épocas, 128 de prueba):

  | Escala | AUC | Sensibilidad | Especificidad | P(error \| error anterior) |
  | --- | --- | --- | --- | --- |
  | 2.0 | 0.982 | 0.944 | 0.902 | 0.53 |
  | 3.0 | 0.975 | 0.932 | 0.886 | 0.60 |
  | **4.0 (elegida)** | **0.971** | 0.933 | 0.865 | 0.63 |
  | 5.0 | 0.968 | 0.930 | 0.860 | 0.65 |

- **El AUC se satura cerca de 0.97,** porque la velocidad y la frecuencia cardíaca informan mucho sobre la apertura: la válvula se cierra sobre ~9 000 rpm en casi todos los escenarios.
- **Se eligió 4.0 a mano.** Da un AUC en el borde del rango, una tasa de error total de ~10% (como la clínica) y errores persistentes (P(error | error anterior) = 0.63). La escala 5 multiplica por cinco los errores publicados casi sin bajar el AUC.
- **Supuesto:** el ruido extra sobre los errores publicados reemplaza fuentes de error que el modelo no tiene, como la variabilidad de la morfología del latido entre pacientes y las arritmias.
- **Los errores salen persistentes sin imponerlos:** es el régimen de errores correlacionados de Li et al., producido por la fisiología (latidos cerca del umbral, error de estimación fijo por paciente).

**Distribución de entrenamiento: corrección tras una prueba de recompensa.**

El primer oracle v2 se entrenó con N = 3, escenarios amplios (Rs hasta 2.5 x) y volemia desde 0.75 x. Se estancó con la válvula abierta en ~86% de los latidos, reflujo en ~50% y gasto de ~3.5 l/min. Se detuvo y se probaron variantes con corridas cortas de oracle (8 M latidos, semilla 0; `scripts/p65_reward_probe.py`, `results/phase6_5/reward_probe.json`):

| Variante | Retorno por latido | Violaciones / 100 | AVO | BACK | Gasto bajo | Gasto |
| --- | --- | --- | --- | --- | --- | --- |
| Amplia, N = 3, w_perf 0.1 | −0.089 | 0.98 | 0.54 | 0.47 | 0.74 | 3.63 |
| Amplia, N = 3, w_perf 1.0 | −0.387 | 3.79 | 0.73 | 0.49 | 0.75 | 3.56 |
| Amplia, N = 5, w_perf 0.1 | −0.081 | 0.58 | 0.41 | 0.43 | 0.73 | 3.68 |
| Amplia, N = 5, w_perf 1.0 | −0.317 | 9.48 | 0.47 | 0.21 | 0.72 | 3.95 |
| Rs ≤ 2.0 x, N = 3 | −0.057 | 0.23 | 0.87 | 0.29 | 0.52 | 4.60 |
| Rs ≤ 2.0 x, N = 3, w_perf 0.3 | −0.107 | 0.30 | 0.87 | 0.30 | 0.52 | 4.57 |
| **Rs ≤ 1.6 x, volemia ≥ 0.80, N = 3 (elegida)** | −0.033 | 0.25 | 0.88 | 0.08 | 0.41 | 4.90 |
| Fases 3–6 (Rs ≤ 1.56 x, volemia ≥ 0.85), N = 3 | −0.030 | 0.16 | 0.90 | 0.06 | 0.34 | 5.11 |

- **El problema no era el peso de la recompensa ni N, sino la hipertensión severa del entrenamiento.** Con Rs sobre ~2 veces la base, cualquier velocidad que mantenga la válvula abriendo produce reflujo y bajo gasto. No existe un punto de operación aceptable, y la política aprende un régimen de bajo gasto.
- Clínicamente, esa hipertensión se trata con fármacos (declaración de la AHA sobre hipertensión en pacientes con VAD), no con la velocidad de la bomba.
- **Se eligió Rs de 0.44 a 1.6 x y volemia de 0.80 a 1.10 x, con N = 3 y w_perf = 0.1.** Esto se desvía del plan, que pedía entrenar con escenarios amplios. La generalización se evalúa con la familia de hipertensión (Rs 1.5–2.0 x, en parte fuera del rango de entrenamiento) y se protege con la capa de seguridad de B4.
- Los datos se recolectaron de nuevo con esta distribución. La calibración del ruido (escala 2.0) se hizo con la distribución amplia; el AUC del detector con la distribución final se reporta en B2.

**Peso del reflujo: deriva del entrenamiento con N = 3.**

Con ruido 4, N = 3 y r_back = −0.1, el oracle semilla 0 estuvo en un régimen sano a los 5–10 M latidos: retorno −0.034 y 10% de reflujo. Desde ahí derivó hasta un régimen de baja velocidad: retorno −0.116, válvula abierta en 99.6%, reflujo en 86% y gasto de 4.0 l/min. **PPO empeoró sostenidamente el retorno de la propia recompensa:** la política se volvió cada vez más conservadora frente a las violaciones.

Pruebas de longitud completa (`scripts/p65_probe_full.py`, detenidas a los 17 M):

| Variante | Reflujo a 7.4 M → 17.2 M | Retorno a 17.2 M | Violaciones / 100 |
| --- | --- | --- | --- |
| Ruido 2, N = 3, r_back −0.1 | 0.075 → 0.345 (deriva) | −0.056 | 0.13 |
| Ruido 4, N = 5, r_back −0.1 | 0.082 → 0.156 (deriva lenta) | −0.037 | 0.07 |
| **Ruido 4, N = 3, r_back −0.3 (elegida)** | 0.072 → 0.063 (estable) | −0.046 (mejorando) | 0.46 |

- **La deriva no la causa el ruido sino el desequilibrio de pesos.** r_back = −0.1 se fijó en la Fase 5 para N = 5. Con N = 3, la presión de violación es mucho mayor y el reflujo queda subvalorado. Las Fases 3–4 usaban −0.5.
- Con −0.3 la política intercambia algo de violaciones (0.46 por 100 latidos) por mucho menos reflujo, y se mantiene estable.
- El oracle semilla 0 del pipeline es la misma corrida que esa prueba (misma semilla y configuración), así que confirma el comportamiento hasta 30 M.

## B2. Datos, detector por latido y modelos de inferencia

**Datos:** entorno v2 final; 512 / 64 / 128 episodios de entrenamiento / validación / prueba, de política aleatoria y del oracle semilla 0 estocástico. Con la política aleatoria la válvula abre en 57% de los latidos; con el oracle, en 87%. Todos los modelos usan exactamente los mismos episodios.

**Detector por latido** (prueba con datos aleatorios y del oracle):

| Detector | AUC | Sensibilidad | Especificidad | P(error \| error anterior) |
| --- | --- | --- | --- | --- |
| CNN (forma de onda + rasgos) | 0.894 | 0.973 | 0.631 | 0.45 |
| **MLP (forma de onda + rasgos, elegido)** | **0.922** | 0.963 | 0.668 | 0.44 |

- **El AUC sobre los datos del pipeline (0.89–0.92) queda bajo el de la calibración (0.971, solo política aleatoria).** Los datos del oracle operan cerca del umbral de apertura, donde detectar es más difícil. La detección queda algo más difícil que la clínica (0.95), con errores persistentes.
- **Se usa el MLP para todos los métodos** (Naive, IBU, IBU-hist, TDM-prop y TDM): es el mejor detector por latido. Una primera pasada usó la CNN por error, se detuvo tras 2 agentes y sus modelos quedaron en `results/phase6_5/tdm_cnn_first/`.

**Calidad de la creencia, primera pasada con la CNN** (log-verosimilitud del estado verdadero; masa en u0 con la válvula cerrada):

| Naive | IBU | IBU-hist | TDM-prop | TDM |
| --- | --- | --- | --- | --- |
| −1.01 / 0.59 | −0.54 / 0.55 | −0.39 / 0.38 | −0.36 / 0.38 | −0.36 / 0.40 |

- **Con la misma información sobre AVO que IBU, TDM-prop ya iguala a TDM completo.** La ventaja frente a IBU viene de usar la historia (modelar los errores persistentes), no del flujo extra, que era justamente la confusión de la Fase 6.
- **IBU-hist** (IBU alimentado con un modelo recurrente de proposiciones) queda cerca: buena parte de la ganancia se obtiene usando historia en el modelo de abstracción, prediga proposiciones o el estado de la RM.

(los resultados con el MLP y los de los agentes se agregan al terminar)

**Calidad de la creencia con el MLP** (log-verosimilitud / masa en u0 con la válvula cerrada): Naive −0.96 / 0.54; IBU −0.51 / 0.51; IBU-hist −0.40 / 0.41; TDM-prop −0.36 / 0.41; TDM −0.36 / 0.41. Es el mismo patrón que con la CNN.

## B3. Agentes y evaluación clínica

**Configuración:**

- 60 agentes en el entorno v2, PPO sin recorte del valor y 30 M latidos cada uno.
- 10 semillas: oracle, IBU, TDM y PPO recurrente. 5 semillas: Naive, IBU-hist, TDM-prop y oracle con presiones.
- Evaluación (`scripts/eval_phase6_5.py`, `results/phase6_5/eval.json`), siempre desde la velocidad del clínico:
  - 1 024 escenarios de la distribución de entrenamiento, con 1 200 latidos (~15 min);
  - las familias de hipertensión, ejercicio e hipovolemia, con 256 escenarios de 900 latidos cada una.
- Los controladores clásicos son deterministas; las modulaciones usan los parámetros ajustados en la Fase 6 sobre v1, sin reajustar.
- Resumen completo en `results/phase6_5/summary.md`.

**Resultados** (IQM sobre semillas; "> 10 s" es el % del tiempo en cierres de más de 10 s):

| Política | n | Aleatorio: > 10 s | Reflujo | Apertura | Retorno | Hipertensión: > 10 s | Reflujo | Hipovolemia: > 10 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | 10 | 0.04% | 4% | 0.85 | −43 | 0.0% | 62% | 0.05% |
| oracle con presiones | 5 | 0.02% | 29% | 0.87 | −131 | 0.0% | 80% | 0.0% |
| Naive | 5 | 6.7% | 36% | 0.92 | −186 | 13.0% | 80% | 4.5% |
| IBU | 10 | 2.8% | 49% | 0.97 | −215 | 6.6% | 87% | 1.6% |
| IBU-hist | 5 | 6.5% | 38% | 0.93 | −189 | 11.0% | 84% | 6.8% |
| TDM-prop | 5 | 6.1% | 44% | 0.94 | −209 | 11.8% | 86% | 4.0% |
| TDM | 10 | 7.2% | 40% | 0.93 | −198 | 16.9% | 78% | 6.0% |
| PPO recurrente | 10 | 16.5% | 32% | 0.82 | −238 | 18.3% | 73% | 20.1% |
| Velocidad del clínico | 1 | 15.0% | 16% | 0.85 | −151 | 56.3% | 54% | 75.6% |
| Modulación en lazo abierto | 1 | 6.9% | 23% | 0.88 | −164 | 0.0% | 63% | 65.1% |
| Regla con el detector MLP | 1 | 68.0% | 2% | 0.30 | −316 | 82.5% | 10% | 82.8% |
| Regla con la etiqueta verdadera | 1 | 0.03% | 4% | 0.92 | −50 | 0.0% | 64% | 0.0% |

**Hipótesis principales** (Welch bilateral, Holm):

| Prueba | Δ media del tiempo cerrado > 10 s | p (Holm) | Resultado |
| --- | --- | --- | --- |
| P1, hipertensión: TDM − IBU | +10.5 puntos | 5·10⁻⁷ | **Rechazada, en la dirección contraria** |
| P1, aleatorio: TDM − IBU | +4.5 puntos | 1·10⁻⁵ | **Rechazada, en la dirección contraria** |
| P2, aleatorio: oracle − PPO recurrente | −22.7 puntos | 0.015 | **Apoyada** |

**Interpretación:**

1. **H1 (estructura de la RM) se sostiene con detección realista:** el oracle tiene ~0% de cierre prolongado y 4% de reflujo, mientras el PPO recurrente sin RM tiene 16.5% y 32%.
2. **H2 en su forma declarada no se sostiene.**
   - Con detección de nivel clínico (AUC ~0.92, errores persistentes), mejor inferencia no da mejor control.
   - La creencia mejora mucho de Naive a TDM (log-verosimilitud de −0.96 a −0.36), pero el control no sigue ese orden. En retorno, Naive (−186) ≈ IBU-hist (−189) ≈ TDM (−198) > TDM-prop (−209) > IBU (−215). En cierre prolongado IBU es el mejor.
   - Lo que más explica el resultado es **cuán conservadora se vuelve la política**. IBU, con su creencia difusa, baja la velocidad y deja la válvula abierta casi siempre (apertura 0.97), pagando con reflujo en la mitad de los latidos. Las creencias más nítidas (TDM, y Naive con sus detecciones duras) llevan a políticas menos conservadoras: menos reflujo, más cierres.
3. **La brecha con el oracle domina todo lo demás.** Todos los agentes con creencia tienen 36–49% de reflujo (el oracle, 4%) y entre 50 y 170 violaciones por hora. Ninguna inferencia aprendida sobre un detector de este nivel basta para un control comparable al oracle.
4. **Contra los controladores clásicos con la misma información, los agentes ganan con holgura:**
   - la regla con el detector MLP se cree sus falsos positivos (especificidad 0.67) y deja la válvula cerrada 68–83% del tiempo;
   - la velocidad del clínico falla en hipertensión (56%) e hipovolemia (76%);
   - la modulación en lazo abierto resiste la hipertensión pero falla en hipovolemia (65%).
5. **Con la etiqueta verdadera, una regla simple iguala al oracle** (0.03%, 4% de reflujo). Toda la dificultad está en la detección.
6. **H3 se invierte respecto de la Fase 6:** el oracle con presiones no es mejor que el de flujo estimado (29% de reflujo contra 4%). Con el estado de la RM, las señales de la bomba bastan.
7. **La succión no aparece en ninguna política aprendida** (0 por hora), como se anticipó en B1.

**Implicación para el informe:** el aporte robusto es H1 junto con la comparación contra controladores clásicos con la misma información. La parte de inferencia queda como un resultado negativo informativo: con errores realistas, mejorar la creencia no basta, y la política compensa la incertidumbre volviéndose conservadora. Esto matiza a Li et al. (2024) en un dominio de control con compromisos entre objetivos.

## B4. Capa de seguridad tipo Simplex

**Montaje** (`lvadrm/safety.py`, `scripts/eval_safety_phase6_5.py`, `results/phase6_5/eval_safety.json`):

- Agentes TDM e IBU, semillas 0–4.
- El control pasa al respaldo cuando:
  - la observación queda fuera de distribución: distancia de Mahalanobis de los rasgos de flujo sobre el cuantil 0.995 del entrenamiento durante 3 latidos;
  - o la creencia es incierta: entropía mayor que 0.8 nats.
- Respaldo: modulación en lazo abierto alrededor de la velocidad del clínico, con permanencia mínima de 30 latidos.
- Conjuntos de prueba:
  - 512 escenarios normales;
  - las tres familias de estrés, con 128 escenarios cada una;
  - 512 escenarios fuera de distribución: Rs 1.8–2.5 x, Emax 0.5–0.7 x y volemia 0.70–0.80 x.
- P3, con el margen fijado antes de evaluar: TDM con la capa no es inferior a la velocidad del clínico fuera de distribución (margen de 2 puntos) y es superior dentro de ella.

**Resultados** (% del tiempo en cierres de más de 10 s; "respaldo" es la fracción de latidos bajo el respaldo):

| Política | Normales | Respaldo | Hipertensión | Hipovolemia | Fuera de distribución | Respaldo |
| --- | --- | --- | --- | --- | --- | --- |
| Velocidad del clínico | 15.1% | — | 54.9% | 75.7% | 61.8% | — |
| Modulación en lazo abierto | 7.0% | — | 0.0% | 65.7% | 60.8% | — |
| Regla con la etiqueta verdadera | 0.0% | — | 0.0% | 0.0% | 60.6% | — |
| TDM | 6.7%* | — | 16.6%* | 5.7%* | 92.1% | — |
| TDM + Simplex | 7.1% | 28% | 12.8% | 29.2% | 89.2% | 3% |
| IBU | 2.6%* | — | 6.4%* | 1.6%* | 89.6% | — |
| IBU + Simplex | 6.7% | 72% | 0.2% | 56.9% | 81.0% | 13% |

\* De `eval.json`: misma distribución, pero otra muestra de escenarios.

**P3:**

| Parte | Comparación | Resultado |
| --- | --- | --- |
| No inferioridad fuera de distribución | 89.2% contra 61.8% + 2 puntos (p unilateral = 1) | **Rechazada** |
| Superioridad dentro de la distribución | 7.1% contra 15.1% (p = 6·10⁻⁵) | **Apoyada** |

**Interpretación:**

1. **El conjunto fuera de distribución es casi infactible.** Incluso la regla con la etiqueta verdadera deja la válvula cerrada 60.6% del tiempo: con contractilidad tan baja, volemia baja y resistencia alta, la válvula no abre ni a la velocidad mínima. Es el mismo problema de ω_min de la Fase 6 (sección B).
   - Ahí cualquier política que no suba la velocidad queda en ~61%.
   - Los agentes, que suben la velocidad para perseguir el gasto, quedan en 89–92%.
   - Es una falla de diseño del conjunto, que no se revisó antes; debió calibrarse su factibilidad.
2. **Los disparadores no detectan este cambio de distribución.**
   - El respaldo actuó solo en 3% (TDM) y 13% (IBU) de los latidos fuera de distribución.
   - Con el flujo estimado (ruido escala 4), la distancia de Mahalanobis de los rasgos no separa los escenarios nuevos.
   - La creencia de TDM sigue confiada cuando está equivocada: es sobreconfianza fuera de distribución.
3. **Dentro de la distribución, la entropía dispara mucho.** El respaldo actúa en 28% de los latidos con TDM y en 72% con IBU, cuya creencia es difusa. Para IBU la capa cambia la política por la modulación: arregla la hipertensión (0.2%) pero empeora la hipovolemia (57%), donde la modulación falla.
4. **Para que sirva haría falta:**
   - disparadores fisiológicos sobre secuencias, por ejemplo latidos consecutivos sin apertura detectada o velocidad persistentemente alta;
   - incertidumbre calibrada, con ensambles;
   - un respaldo que funcione en hipovolemia;
   - un conjunto fuera de distribución factible.

## B5. Resumen estadístico

| Hipótesis | Resultado |
| --- | --- |
| P1, TDM con menos cierre prolongado que IBU | **Rechazada:** significativa en la dirección contraria (hipertensión y aleatorio, p con Holm ≤ 1·10⁻⁵) |
| P2, oracle contra PPO recurrente | **Apoyada** (p con Holm = 0.015) |
| P3, Simplex | **Rechazada** en no inferioridad fuera de distribución (conjunto casi infactible); **apoyada** en superioridad dentro de la distribución |

El detalle con IQM e intervalos bootstrap está en `results/phase6_5/summary.md` y `analysis.json`.
