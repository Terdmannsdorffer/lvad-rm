# Fase 7: escalera de información (rediseño tras el feedback del profesor)

Feedback del 9 de noviembre de 2026. Las hipótesis y métricas de la sección 2 se escribieron el 23 de noviembre, antes de correr los controladores y los agentes.

## 1. Diseño

| Tema | Decisión | Justificación |
| --- | --- | --- |
| Paciente base | Paciente con insuficiencia con Emax × 0.7 (`WEAK_PATIENT`) | El objetivo de 5 l/min no se alcanza con la válvula abriendo en cada latido: en ωAVO el gasto es 4.33 l/min. Decisión del autor |
| RM | Saturada: se queda en u(N−1) y paga r_v en cada latido adicional cerrado | Con vuelta a u0, no abrir nunca paga r_v/N por latido y puede ser lo óptimo. Decisión del autor |
| N | 10 principal; 5 y 20 de sensibilidad | El patrón de referencia gana +5 %, +16 % y +40 % de gasto; con N = 3 gana solo +1 % |
| Pacientes | "fijo" (paciente base) y "desconocido" (Emax ± 25 %, Rs 0.8–1.3 ×, HR 65–100, volemia 0.88–1.05 ×, hasta 3 perturbaciones) | 91 % de los pacientes desconocidos necesita contar y en todos hay ventana sin reflujo |
| Observación | Flujo medio estimado, velocidad, HR y últimas 4 acciones, más la detección del latido | Chequeo de fuga, sección 3 |
| Detector | d = 1[T + b + ε > τ], τ = 56.2 ms, σ = 36.5 ms; λ = fracción de la varianza en b (por episodio) | Feedback 3.3. Reproduce 86 % / 93 % de Clifford et al. (2019) para todo λ, sobre los datos de calibración |
| Estimadores | Naive, IBU (con P(AVO\|d) y con clasificador por latido), IBU-hist, TDM y filtro sobre (umbral del paciente, sesgo, contador) | Todos reciben la misma entrada por latido. El filtro conoce λ (hay una variante con λ = 0.5 supuesto) |
| Datos de entrenamiento | Solo de políticas que no ven la etiqueta | Una política que reacciona a la etiqueta la delata con sus acciones |
| Recompensa | −r_v por violación, −0.3 por reflujo > 1 ml, −\|CO − 5\|/5, −0.01 por cambio relativo de velocidad; r_v ∈ {0.3, 1, 3} | Curvas y no puntos |
| Inicio | Velocidad poblacional ± 5 %; se descartan 30 latidos | La velocidad inicial del clínico revela el umbral del paciente |

## 2. Hipótesis y métricas (antes de correr)

Ejes: violaciones (% de latidos en violación, N = 10) y desvío del gasto (|CO − 5|/5), en 512 escenarios de prueba con los mismos escenarios y semillas para todos.

- **H1, contar importa.** El mejor patrón periódico supera en gasto a la mejor velocidad constante que siempre abre. Sin RL (mapa y patrón de referencia).
- **H2, la realimentación importa.** El patrón a ciegas iguala al contador verdadero con el paciente fijo, y no con el paciente desconocido. Se compara el desvío mínimo de gasto con violaciones ≤ 1 %.
- **H3, el estimador importa.** Predicción revisada tras la teoría: la diferencia entre IBU y los estimadores que usan historia **no** desaparece con λ = 0, porque la fisiología persiste; con λ alto el promedio de IBU mejora y el daño se concentra en pocos pacientes.
- **Métricas secundarias:** violaciones en el peor 5 % de los pacientes, calibración de P(contador = N−1), reflujo.
- **Dos lecturas de cada comparación:** (a) mismo controlador (monitor) con distintos estimadores, sin RL; (b) agentes PPO con tres valores de r_v.

## 3. Calibración y chequeo de fuga

Latidos con velocidad a menos de 10 % del umbral del paciente y política a ciegas; regresión logística (exactitud balanceada / AUC):

| Entrada de un latido | Exactitud balanceada | AUC |
| --- | --- | --- |
| Observación reducida (la de la política) | 0.77 | 0.84 |
| + mínimo, máximo y desviación del flujo | 0.85 | 0.93 |
| Detección sola | 0.88 | 0.88 |
| Observación reducida + detección | 0.89 | 0.94 |

Con todos los latidos, la observación reducida llega a AUC 0.98: lejos del umbral la velocidad basta. El detector aporta cerca del umbral, que es donde se opera.

Duración de la apertura (percentiles 5/50/95): 62 / 104 / 223 ms. Paso de integración: al dividir dt por dos cambian 8 de cada millón de etiquetas AVO.

## 4. Desviaciones respecto de la sección 2 (24 de noviembre)

Todas posteriores a ver algún resultado. Las 1, 2, 4 y 5 se decidieron mirando solo las referencias (constante, a ciegas, contador verdadero, oracle PPO). **Las 3 y 6 se decidieron después de ver resultados de estimadores:** la 3, tras comparar la calidad de la creencia del filtro con la de TDM; la 6, tras ver en el test que la configuración elegida para IBU-hist violaba mucho más que en validación. Por eso el filtro no se presenta como un método confirmado de antemano, y las curvas completas (que no dependen de la elección en validación) se reportan junto a la tabla.

1. **Grilla de ajuste más fina** (consigna de flujo cada 0.2 l/min en vez de 0.5). Con la grilla gruesa el contador verdadero tenía un solo punto en su curva.
2. **Segunda versión de la velocidad nominal ("rampa").** El patrón a ciegas con seguimiento de flujo hereda el error del sensor, así que no era el mejor patrón a ciegas posible: con el paciente fijo quedaba peor que la velocidad constante. La versión "rampa" fija la velocidad nominal como un factor de la ωAVO medida al inicio. Es un baseline más duro para H2. Se reportan las dos.
3. **Filtro:** la deriva, las tablas y la grilla del sesgo se ajustaron con los datos de calibración (`scripts/phase7_tune_filter.py`), después de ver que el filtro inicial quedaba peor que TDM en datos retenidos.
4. **PPO:** γ = 0.95, entropía 0.003 y 20 M latidos, elegidos con una corrida del oracle (γ = 0.99: recompensa −0.178; 0.97: −0.162; 0.95: −0.159 a los 10 M). Con γ = 0.99 el oracle quedaba peor que la regla con contador verdadero (desvío 0.178 contra 0.172).
5. **Agentes recurrentes con estimador** (`true_rnn`): el oracle sin memoria no puede recordar el umbral del paciente. Es la comparación con igual memoria e igual información que pide el feedback.
6. **Segunda validación** con 512 episodios para elegir entre las configuraciones de Pareto (`scripts/phase7_revalidate.py`): con 64 episodios y violaciones de cola pesada, la elección era ruidosa. No usa los episodios de prueba.
7. **Más agentes PPO que los planificados** (`scripts/phase7_queue_ext.sh`): agentes que además observan presiones (sEDP y LVPmax − AOPmax de los dos últimos latidos) bajo la misma recompensa y sin detector, con y sin memoria, que reemplazan a la réplica del MDP de [1] como pide el feedback (3.4); IBU-hist con N = 10, y una escalera reducida con N = 20 (true, window, last; con λ = 0.5: naive, IBU-hist, TDM, filtro, last). Se decidió después de ver la semilla 0 con N = 10 (etiquetas verdaderas y λ = 0.5). IBU-hist no estaba entre los agentes aunque es la variante de la identidad teórica. N = 20 pone a prueba si la ventaja de la RM sobre la memoria aparece con un N mayor, como pide el feedback (3.2), porque con N = 10 no apareció. Los hiperparámetros no cambian.

## 5. Diagnóstico de holgura (contador verdadero, paciente fijo, seguimiento de flujo)

Desvío mínimo con ≤ 1 % de violaciones según la escala del error de flujo: 0.110 (escala 1, la usada), 0.091 (0.3) y 0.090 (sin error); la ganancia del seguimiento no influye. El sensor de flujo esconde unos 0.02. El patrón de referencia del mapa, que no usa el sensor, llega a 0.055.

## 6. Datos reales, mecanismo y calibración (24 de noviembre, después de los resultados de la sección 4)

**Predicciones escritas antes de la primera corrida con datos reales**. Datos: solo anotaciones de PhysioNet; el detector es un clasificador del intervalo RR y de su razón con el anterior; división por registro.

- P1 (FA): IBU sin memoria subestima P(u ≥ k), más con k mayor.
- P2 (FA): IBU con clasificador recurrente también subestima, pero menos.
- P3 (FA): TDM queda calibrado (razón 0.9–1.1) en la misma base; en la base externa, quizá no.
- P4 (ventriculares): sin signo a priori; el signo de β₂ (covarianzas a un paso, en los registros de entrenamiento) predice el de β_k para k = 3 a 5.
- P5: los errores del detector tienen una parte persistente por registro.

**Qué se vio antes de fijar la versión final.** Una corrida piloto de una división (FA) y dos (ventriculares) con los mismos modelos. Después se agregaron el clasificador de 16 latidos (la primera versión se consideró un adversario débil), cinco divisiones por registro y un script reproducible (`scripts/phase7_realdata.py`). No se cambió ninguna hipótesis ni métrica.

**Mecanismo** (`scripts/phase7_mechanism.py`): violaciones contra el sesgo del detector, en 512 episodios de otra semilla. Es exploratorio: se hizo después de ver que el 5 % peor concentra las violaciones.

**Calibración por paciente** (`scripts/phase7_calibrated.py`): el sesgo residual s_b (error de la calibración más deriva) entra como una fracción λ' menor. Es exploratorio, pero el diseño (grilla, validación en dos pasos, 512 episodios de prueba) es el mismo que el de los demás controladores. Supone que el sesgo no depende del paciente y no cambia dentro del episodio; la deriva queda dentro de s_b.

**Resultado de las predicciones** (`results/phase7/realdata_*.json`, cinco divisiones por registro; intervalos por bootstrap de registros):

| Predicción | Resultado |
| --- | --- |
| P1 (FA, IBU sin memoria subestima, más con k) | Se cumple: razón 0.03 a 30 latidos en ambas bases, intervalo negativo en 10 de 10 |
| P2 (FA, IBU recurrente subestima menos) | Se cumple en parte: 0.82 y 0.81 a 30 latidos, intervalo negativo en 3 de 5 (misma base) y 5 de 5 (otra base) |
| P3 (FA, TDM calibrado) | Se cumple: 1.04 (0.99–1.10) en la misma base y 1.00 (0.98–1.03) en la otra, que no era lo esperado |
| P4 (ventriculares, el signo de β₂ predice k = 3 a 5) | Se cumple para IBU sin memoria (negativo en 5 de 5). Para la ventana y el recurrente, solo 2 y 3 de 5 son significativos y en una división el signo es el contrario |
| P5 (errores persistentes) | Se cumple: correlación latente de 0.12 a 0.72 |

No predicho: TDM sobreestima las rachas ventriculares raras (3.6× a 3 latidos, 8× a 5), porque en entrenamiento son más frecuentes que en prueba (0.4–0.5 % contra 0.02–0.23 % de los latidos).

**Calibración por paciente** (nominal "rampa", configuración elegida en validación): con λ = 0.9 y error residual ≤ 10 ms, desvío 18.3 % (contador verdadero 17.8, a ciegas 20.9, TDM 19.3) con 0.9–1.1 % de violaciones; con 20 ms, 22.0 %, peor que el patrón a ciegas. Con λ = 0.5, 19.9 % (TDM 18.5). **Mecanismo** (rampa, λ = 0.9): todas las violaciones de Naive e IBU están en pacientes con sesgo positivo del detector.

## 7. Robustez de "calibrar o recordar" (diseño y predicciones del 24 de noviembre, antes de la primera corrida)

Una lectura crítica del borrador para CinC señaló que la conclusión descansa en dos valores de λ, que la calibración no envejece, que los pacientes de prueba salen de la misma distribución y que falta un baseline simple de memoria. Antes de escribir esto se habían visto los resultados de las secciones 4 a 6 (λ = 0, 0.5 y 0.9; calibración con λ = 0.5 y 0.9) y que en las cinco poblaciones desplazadas la válvula puede abrir dentro del rango de velocidades; nada de lo que sigue. Script: `scripts/phase7_robust.py` (cola `scripts/phase7_robust.sh`).

**Protocolo común.** N = 10, velocidad nominal "rampa", misma clase de controlador, 64 episodios para la grilla, 512 para elegir (menor desvío con ≤ 1 % de violaciones), 512 de prueba. En `shift` y `mismatch` no se vuelve a ajustar nada. La comparación principal es el desvío pareado en la prueba entre la configuración elegida de cada política, con intervalos por bootstrap de pacientes; si ninguna configuración cumple el 1 % se dice.

| Parte | Diseño |
| --- | --- |
| naive / learned | λ = 0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. Naive sin calibrar y con sesgo residual 0, 10 y 20 ms; TDM e IBU-hist entrenados para cada λ. Los residuos mayores que la desviación del sesgo (√λ σ) no se corren: medir empeoraría |
| drift | λ = 0.9. El sesgo cambia dentro del episodio con autocorrelación exp(−t/θ) y la misma varianza (las tasas por latido no cambian): deriva lenta (θ = 10000, 2500, 800 y 250 latidos) o cambios abruptos (θ = 2500 y 800). Calibración al inicio con error de 5 ms (supuesto); además, recalibración cada 100 latidos (θ = 800 y 250 lento, 800 abrupto). TDM sin calibrar y sin volver a entrenar con θ = 2500 y 250 lento y 800 abrupto |
| combine | TDM entrenado para el detector calibrado: (λ, residuo) = (0.9, 10), (0.9, 20) y (0.5, 10) |
| vote | Naive con antirrebote: una apertura cuenta si al menos 2 de los últimos 2 (o 3) latidos se detectan abiertos. Sin calibrar con λ = 0, 0.5 y 0.9; calibrado con (0.5, 0), (0.5, 10) y (0.9, 10) |
| window | Red sin memoria que predice el contador con los últimos 16 latidos, entrenada con los mismos episodios que TDM; λ = 0, 0.5 y 0.9 |
| seeds | TDM e IBU-hist con otras dos semillas de datos y entrenamiento; λ = 0.5 y 0.9 |
| mismatch | TDM e IBU-hist entrenados con un λ y usados con otro (0, 0.5, 0.9), con la configuración elegida para el λ de entrenamiento |
| shift | Pacientes fuera del rango de entrenamiento: Emax 0.55–0.75 ×, Rs 1.3–1.6 ×, HR 100–125, volemia 0.80–0.88 ×, y todos los rangos ampliados a la vez (Emax 0.6–1.4, Rs 0.65–1.6, HR 55–125, volemia 0.82–1.10). Supuestos: los rangos no vienen de una población clínica |

**Predicciones.**

- R1: el desvío de TDM y el de Naive calibrado a 10 ms se cruzan entre λ = 0.7 y 0.9; con calibración exacta, entre 0.6 y 0.8. Naive sin calibrar deja de cumplir el 1 % desde λ = 0.6.
- R2: con deriva lenta decide la raíz del error cuadrático medio del sesgo residual durante el episodio (8.5, 14.4, 23.1 y 35.0 ms para θ = 10000, 2500, 800 y 250): con 10000 se conserva la ventaja (desvío ≤ 18.6 %) y con 800 y 250 se pierde contra el patrón a ciegas. Recalibrar cada 100 latidos (13.0 ms) la recupera con θ = 800 y no con 250 (21.1 ms). Los cambios abruptos dejan más violaciones en el peor 5 % que la deriva lenta con el mismo θ. TDM queda a ± 0.5 puntos de su resultado con sesgo constante.
- R3: TDM sobre el detector calibrado no es peor que ninguno de los dos por separado; con (0.9, 20) baja del 20 % de desvío, donde la calibración sola da 22.0 %.
- R4: el orden contador verdadero < Naive calibrado < a ciegas en desvío se mantiene en las cinco poblaciones. El presupuesto de 1 % no se mantiene para TDM e IBU-hist con HR alta y con Emax bajo, y empeoran más que la regla calibrada.
- R5: la red de 16 latidos queda a menos de 0.5 puntos de TDM con λ = 0 y peor con 0.9. El antirrebote mejora a Naive con λ = 0 y 0.5 y no con 0.9 sin calibrar.
- R6: entrenar con λ = 0 y usar con 0.9 lleva a más de 2 % de violaciones; entrenar con 0.9 y usar con 0 o 0.5 queda a menos de 0.5 puntos de desvío de lo entrenado para ese λ.
- R7: el desvío cambia menos de 0.4 puntos entre semillas de entrenamiento.

**Qué no se adoptó de esa lectura.** Recalibración adaptativa (requiere un modelo de la medición ecocardiográfica que no tenemos) y afirmar validez clínica (el trabajo es un estudio en simulación y así se escribe). El sesgo dependiente del paciente queda como limitación declarada.
