# Fase 6.5: validez interna y relevancia clínica (plan)

Fase intermedia entre la Fase 6 (robustez) y la Fase 7 (informe). Ataca las debilidades de los resultados de la Fase 6, sección H. Este documento es el plan; los resultados irán en `phase6_5_decisions.md`. Las hipótesis principales de la sección 6 quedan declaradas aquí antes de entrenar.

## 1. Objetivo

Convertir los resultados en evidencia defendible:

1. **Una prueba justa de H2:** mismas entradas y mismos datos para todos los métodos de inferencia, como en Li et al. (2024).
2. **Un entorno coherente con la clínica:** detección de la apertura con dificultad realista, recompensa fisiológica y succión alcanzable.
3. **Métricas con significado clínico,** en segundos y en razón de apertura.
4. **Seguridad fuera de distribución.**
5. **Estadística declarada de antemano.**

## 2. Mapa: problema → solución

| # | Problema (crítica de la Fase 6) | Solución | Bloque |
| --- | --- | --- | --- |
| 1 | H2 confundida: TDM ve el flujo (97% de exactitud para AVO por sí solo) y Naive e IBU solo ven el detector. TDM v2 además se entrenó con más datos que el detector | Diseño factorial con entradas y datos idénticos, más ablaciones que separan "modelar el estado de la RM" de "usar historia" | B2, B3 |
| 2 | Incoherencia: en el simulador la apertura es casi trivial de ver, pero suponemos un detector clínico de 86/93% | Observación realista (flujo estimado) calibrada para que el mejor detector aprendido tenga desempeño clínico. Los errores emergen del simulador | B1 |
| 3 | Efectos chicos; una violación son ~4 s, y el daño clínico viene de cierres sostenidos | Métricas en segundos y razón de apertura, episodios largos y restricción revisada con el profesor | B0, B1 |
| 4 | El retorno lo domina seguir un gasto fijo de 5 l/min, incluso en ejercicio | Objetivo de flujo tipo Starling (dependiente de la precarga) y análisis de sensibilidad | B0, B1 |
| 5 | La succión nunca ocurre | Escenarios de caída de precarga calibrados para que sea alcanzable | B1 |
| 6 | Fragilidad: un detalle de PPO cambió conclusiones; 3–5 semillas; muchas comparaciones; la hipertensión es un solo escenario | Hipótesis declaradas de antemano, 10 semillas en las principales, IQM con bootstrap, Holm y familias aleatorias de escenarios de estrés | B5 |
| 7 | Fuera de rango, los agentes con creencia son peores que la velocidad constante | Entrenamiento con escenarios amplios y capa de seguridad tipo Simplex con respaldo clásico | B3, B4 |
| 8 | Modelo 0D sin ventrículo derecho ni barorreflejo; bomba axial | Fuera de alcance; se declara como limitación (ruta al paper) | — |

## 3. Evidencia que guía el diseño

**Criterio clínico de apertura aórtica:**

- **La meta habitual es la apertura intermitente,** no un número de latidos.
  - El protocolo VAFRACT (NCT03937570) define la velocidad óptima como la que permite apertura aórtica intermitente y septum neutro.
  - Una revisión de 2025 sobre trombosis de la raíz aórtica dice que la práctica apunta a la apertura intermitente, y liga el cierre persistente a trombosis de la raíz, insuficiencia aórtica progresiva y embolias, aunque con pocos datos prospectivos.
- **La razón 1:3** (al menos una apertura cada tres sístoles) aparece como meta de trabajo, pero la única fuente que encontramos es descriptiva, no una guía.
- **La actualización 2024 de la ASE** dejó la apertura como objetivo secundario frente a la remisión de síntomas, pero reconoce el riesgo de trombo en la raíz y de insuficiencia aórtica con la válvula cerrada. Solo leímos un resumen; hay que verificarlo en el documento completo.
- **En 18 pacientes con HVAD,** la mediana de la frecuencia de apertura fue 21.5% de los latidos y la de la duración, 124 ms (Clifford et al. 2019). Nuestros agentes abren en 64–90% de los latidos.
- **El cierre persistente predice la insuficiencia aórtica de novo,** que llega a ser moderada o mayor en 29.7% de 219 pacientes. Ningún estudio da un umbral de duración del cierre: la evidencia mide tiempo total de soporte, no tiempo de cierre.
- **Implicación:** la restricción debe expresarse como razón de apertura o tiempo máximo sin abrir. N = 3 equivale a garantizar la razón 1:3. La motivación del informe debe reconocer que la apertura es un objetivo secundario según la ASE.

**Recompensa fisiológica:**

- **Los controladores tipo Starling** hacen que el flujo medio de la bomba siga la presión de fin de diástole, como la ley de Starling del corazón nativo (Mansouri et al. 2017, in vitro).
- **Frente a la velocidad constante:**
  - aumentan el gasto en ejercicio;
  - mantienen mejor margen contra la succión en hemorragia (presión de fin de diástole de 2.7 contra −0.1 mmHg);
  - sostienen la presión y el flujo cuando cae la contractilidad.
- **Implicación:** un objetivo de flujo que dependa de la precarga es más defendible que 5 l/min fijo.

**Realismo de la observación:**

- **En la práctica el flujo se estima** a partir de la corriente y la velocidad del motor, con una viscosidad preestablecida. En bombas axiales la relación corriente–flujo es no lineal y menos precisa.
- **Errores reportados del flujo estimado:**
  - HVAD en banco: diferencia media de 0.1 ± 0.3 l/min y dispersión de la forma de onda de ~0.7 l/min;
  - Karantonis et al. (2007): 5.77% de error medio.
- **Implicación:** la observación debe usar flujo estimado con sesgo por paciente y error en la forma de onda.

**Succión:**

- **Es un problema de precarga:** la causan la hipovolemia o hemorragia, la falla del ventrículo derecho, las arritmias y la embolia pulmonar.
- **En ovejas, la oclusión de la arteria pulmonar produjo succión en 5–10 latidos** (Reesink et al. 2007). Su persistencia dependió de la resistencia pulmonar, la volemia y la contractilidad del ventrículo derecho; bajar la velocidad no la revirtió, detener la bomba sí.
- **Implicación:** se necesitan escenarios de hipovolemia. Nuestro modelo no tiene ventrículo derecho, así que no puede representar colapso sostenido; es una limitación a declarar.

**Comparación justa de métodos de inferencia** (Li et al. 2024):

- Todos los modelos de abstracción reciben historias de observación y se entrenan con datos equivalentes "ensuring a fair comparison".
- Naive e IBU predicen proposiciones; TDM predice el estado de la RM.
- Se usan 8 corridas por configuración y la log-verosimilitud del estado verdadero de la RM.

**Estadística en RL:**

- Henderson et al. (2018) encontraron que los artículos usaban 5 semillas o menos.
- Colas et al. (2018) muestran cómo el número de semillas controla los errores de tipo I y II (test de Welch y bootstrap).
- Agarwal et al. (2021) recomiendan:
  - la media intercuartil (IQM) con intervalos bootstrap estratificados al 95%;
  - la probabilidad de mejora;
  - con 3 corridas, los intervalos bootstrap quedan demasiado estrechos.

**Seguridad:**

- **La arquitectura Simplex** pasa el control de un controlador avanzado no verificado a uno de respaldo verificado cuando un módulo de decisión lo indica. Neural Simplex agrega la vuelta al controlador avanzado y la probó en un páncreas artificial (Phan et al. 2020).
- **En un banco de pruebas de diabetes,** agentes de RL seguro que cumplían las restricciones en entrenamiento fallaron en pacientes no vistos. Filtrar las acciones con un escudo recuperó la seguridad (Kwon et al. 2026).

## 4. Bloques de trabajo

### B0. Decisiones previas (con el profesor; antes de programar)

| Decisión | Opciones | Recomendación |
| --- | --- | --- |
| D1. Restricción de apertura | (a) N = 5 latidos, como ahora; (b) **N = 3**, que garantiza la razón clínica 1:3; (c) RM temporizada: al menos una apertura cada T segundos (arXiv 2512.17637) | (b) como principal y N = 5 y 10 como sensibilidad; reportar todo en segundos |
| D2. Término de desempeño | (a) **Starling:** objetivo de flujo función de la presión de fin de diástole; (b) gasto según la demanda del escenario; (c) gasto fijo con barrido de w_perf | (a), con barrido de w_perf ∈ {0.03, 0.1, 0.3} como sensibilidad |
| D3. Observación | Flujo estimado con error calibrado a la detección clínica (AUC ≈ 0.95) | Sí |
| D4. Escenarios | Agregar hipovolemia (succión alcanzable) y entrenar con escenarios amplios | Sí |
| D5. Semillas y cómputo | 10 semillas en las comparaciones principales y 5 en las secundarias; ~35 h de GPU | Sí |

### B1. Entorno v2 (4–5 días, poca GPU)

1. **Observación realista** (`EnvConfig.obs_model = "estimated"`):
   - flujo estimado = flujo real × (1 + error relativo por paciente) + sesgo por paciente + ruido de forma de onda, más un filtro pasa bajo y submuestreo;
   - rangos iniciales: sesgo de ±0.3 l/min, error relativo de ~6% y ruido de forma de onda de ~0.7 l/min, según los errores reportados;
   - calibración: se ajusta un solo factor de escala del ruido hasta que el mejor detector aprendido (MLP o CNN sobre la forma de onda) llegue a un AUC de 0.93–0.97 y a sensibilidad y especificidad cerca de 86% y 93%, como en Clifford et al. 2019. No se hace ni trivial ni imposible.
   - Consecuencia: los errores del detector dependen de la fisiología (latidos cerca del umbral, baja pulsatilidad) y se correlacionan en el tiempo porque el estado fisiológico persiste. Es el escenario de Li et al., sin errores impuestos. Los detectores sintéticos quedan solo como análisis secundario.
2. **Recompensa** (según D2):
   - Q_obj = f(presión de fin de diástole), una curva tipo Starling. Se calibra para que el paciente base en reposo quede en ~5 l/min y el gasto suba en ejercicio.
   - Se mantienen la RM y las penalizaciones de reflujo y succión.
   - Se reporta la descomposición del retorno en todos los resultados.
3. **Restricción** (según D1): N como parámetro; métricas en segundos.
4. **Succión alcanzable:**
   - eventos de hipovolemia (Vtot −10 a −25%) y caídas de precarga;
   - con `window.py` se elige el rango para que la succión sea alcanzable dentro de [ω_min, ω_max] en 20–40% de los escenarios de entrenamiento.
5. **Familias de escenarios de estrés** para evaluar, con parámetros aleatorios (~256 escenarios cada una), en vez de un protocolo determinista:
   - hipertensión: Rs × 1.5–2.0, con inicio, rampa, frecuencia y contractilidad aleatorios;
   - ejercicio;
   - hipovolemia;
   - eventos combinados.

   Los tres protocolos actuales se mantienen como ilustración.
6. **Episodios largos de evaluación:** 1 800 latidos (~20–30 min) con varios eventos. El entrenamiento sigue con 300 latidos.
7. **Métricas clínicas:**
   - razón de apertura;
   - duración máxima de cierre (s);
   - % del tiempo en cierres de más de 10, 30 y 60 s;
   - succiones por hora;
   - fracción con reflujo;
   - tiempo con presión de aurícula izquierda > 20 mmHg (congestión) y con gasto bajo;
   - cambios de velocidad por minuto.

**Criterios de aceptación de B1:**

| Criterio | Objetivo |
| --- | --- |
| AUC del mejor detector de AVO sobre la observación v2 | 0.93–0.97 |
| Escenarios de entrenamiento con succión alcanzable | 20–40% |
| Gasto del paciente base en reposo con el objetivo Starling | 4.5–5.5 l/min |
| El objetivo Starling sube en ejercicio | > +15% en el protocolo de ejercicio |
| Velocidad constante del clínico: tiempo con cierre de más de 30 s en la familia de hipertensión | > 0 (la tarea no es trivial) |
| Rendimiento del entorno | ≥ 30 000 latidos/s en la 3090 |

### B2. Datos y modelos de inferencia con entradas equivalentes (2 días, ~4 h de GPU)

- **Datos:** políticas aleatorias y exploratorias en escenarios amplios, más datos de una política oracle preliminar. Todos los modelos usan exactamente los mismos episodios de entrenamiento, validación y prueba.
- **Diseño factorial** (todas las variantes reciben la misma observación v2 por latido):

| Variante | Modelo de abstracción | Predice | Separa |
| --- | --- | --- | --- |
| Naive | Clasificador por latido | Proposiciones (AVO, SUC) | Línea base |
| IBU | Clasificador por latido | Proposiciones | Línea base de Li et al. |
| IBU-hist | Clasificador recurrente (historia) | Proposiciones | ¿Basta con usar historia? |
| TDM-prop | GRU que solo recibe la salida del clasificador por latido | Estado de la RM | ¿Ayuda modelar el estado de la RM con la misma información que IBU? |
| TDM | GRU sobre la observación completa | Estado de la RM | Método propuesto |

- **Métricas de creencia:** log-verosimilitud del estado verdadero, calibración (ECE) y masa en u0 con la válvula cerrada, por familia de escenarios.

### B3. Agentes (4 días, ~22 h de GPU)

- **Entrenamiento:** escenarios amplios con eventos de succión (aleatorización de dominio), PPO sin recorte del valor y los hiperparámetros actuales.
- **Principales (10 semillas):** oracle, PPO recurrente, IBU y TDM.
- **Secundarios (5 semillas):** Naive, IBU-hist, TDM-prop y oracle con presiones (H3).
- **Líneas base:**
  - velocidad constante del clínico;
  - modulación en lazo abierto;
  - regla con el clasificador por latido en el lazo, reajustada con 512 escenarios de validación;
  - controlador inspirado en Simaan, declarado como "inspirado en".
- **Cómputo:** 60 corridas, ~13 h de entrenamiento. Evaluación: episodios largos (~8 min por agente, porque el costo lo domina la latencia secuencial por latido) más las familias, ~10 h.

### B4. Seguridad fuera de distribución (2 días, ~4 h de GPU, sin entrenar)

- **Envoltura tipo Simplex** sobre el mejor agente. Un monitor activa el respaldo con tiempo mínimo de permanencia y devuelve el control cuando se normaliza. Disparadores:
  1. observación fuera de distribución: distancia de Mahalanobis de los rasgos frente a los datos de entrenamiento;
  2. creencia incierta o inconsistente: entropía alta o desacuerdo entre TDM e IBU;
  3. envolvente fisiológica a partir de señales de la bomba: pulsatilidad baja sostenida, reflujo o firma de succión.
- **Respaldos candidatos:** modulación en lazo abierto (0 violaciones en hipertensión en la Fase 6) y velocidad constante del clínico.
- **Evaluación:**
  - costo dentro de la distribución;
  - beneficio fuera de ella: escenarios amplios extremos, contractilidad reducida pero factible e hipertensión extrema.

### B5. Estadística

- **Hipótesis principales** (corrección de Holm sobre P1–P3, α = 0.05; métrica principal: % del tiempo en cierres de más de 10 s, en episodios largos):
  - **P1 (H2 justa):** TDM tiene menos tiempo de cierre prolongado que IBU, con las mismas entradas y datos, en la familia de hipertensión y en los escenarios de prueba aleatorios (Welch bilateral, 10 semillas).
  - **P2 (H1):** el oracle tiene menos tiempo de cierre prolongado que el PPO recurrente.
  - **P3 (seguridad):** TDM con la envoltura Simplex no es inferior a la velocidad constante del clínico fuera de distribución (margen δ por fijar en B0) y es superior dentro de ella.
- **Secundarias:** sin test de hipótesis, o con Holm aparte. Incluyen:
  - TDM-prop contra IBU, que muestra el mecanismo;
  - IBU-hist;
  - los barridos de N y de w_perf;
  - los detectores sintéticos.
- **Reporte:**
  - IQM con intervalos bootstrap estratificados al 95% (rliable);
  - probabilidad de mejora;
  - tamaños de efecto en unidades clínicas (segundos de cierre por hora).
- **Potencia:** con 10 semillas por grupo, el test de Welch detecta efectos de d ≈ 1.3 con potencia de 0.8. Con la dispersión entre semillas de la Fase 6 (desviación de 0.02–0.05 violaciones cada 100 latidos), eso equivale a diferencias de ~0.03–0.07. Supuesto: la dispersión en el entorno v2 es similar; se verifica tras las primeras 5 semillas.

## 5. Calendario y cómputo

| Bloque | Duración | GPU |
| --- | --- | --- |
| B0, decisiones con el profesor | 1–2 días | — |
| B1, entorno v2 y calibración | 4–5 días | ~3 h |
| B2, datos e inferencia | 2 días | ~4 h |
| B3, agentes | 4 días (noches de GPU) | ~23 h |
| B4, seguridad | 2 días | ~4 h |
| B5, análisis y checkpoint | 1–2 días | — |
| **Total** | **~2.5–3 semanas** (13 oct – ~3 nov) | **~35 h** |

La Fase 7 (informe y presentación) quedaría del 3 al 29 de noviembre, con la entrega supuesta la semana del 30 de noviembre.

## 6. Riesgos y planes B

| Riesgo | Señal | Plan B |
| --- | --- | --- |
| La observación calibrada vuelve el problema demasiado difícil | Los agentes con creencia quedan peor que la velocidad constante dentro de la distribución | Usar el extremo superior del AUC plausible (0.97) y reportar la curva desempeño–AUC |
| La recompensa Starling desestabiliza el entrenamiento | Retorno plano o colapsos | Gasto fijo con barrido de w_perf (D2-c) |
| La succión domina los episodios | > 40% de escenarios con succión | Limitar la proporción o la magnitud de la hipovolemia |
| TDM no supera a IBU en la comparación justa | P1 no significativa | Reformular H2 según dónde ayuda modelar el estado de la RM (TDM-prop, familias). Sigue siendo un resultado publicable |
| Falta tiempo | B3 sin terminar el 31 de octubre | Bajar a 5 semillas en las secundarias y eliminar IBU-hist y oracle con presiones; mantener 10 semillas en P1 y P2 |

## 7. Fuera de alcance (se declara)

- Modelo 0D sin ventrículo derecho ni barorreflejo: sin colapso sostenido por succión.
- Bomba axial tipo HeartMate II; los dispositivos actuales son centrífugos.
- Sin validación in vitro.
- La detección por fluctuaciones de la velocidad (el método de Clifford) no se modela, porque la velocidad está idealmente controlada.
- Los resultados de las Fases 5 y 6 se conservan como preliminares (v1); el informe usará los de la Fase 6.5.

## 8. Referencias

- Li, Chen, Klassen, Vaezipoor, Toro Icarte y McIlraith (2024), NeurIPS. https://arxiv.org/abs/2406.00120
- VAFRACT, NCT03937570. https://clinicaltrials.gov/study/NCT03937570
- Revisión de 2025 sobre trombosis de la raíz aórtica tras LVAD. https://pmc.ncbi.nlm.nih.gov/articles/PMC13002484/
- ASE 2024, imagen multimodal en LVAD (solo resumen leído). https://www.jcvaonline.com/article/S1053-0770(25)00371-4/fulltext
- Clifford et al. (2019), Artif Organs 43(5):458–466. https://eprints.victorchang.edu.au/770
- Insuficiencia aórtica de novo y cierre persistente (219 pacientes). https://link.springer.com/article/10.1007/s10047-020-01234-4
- Mansouri et al. (2017), Preload-based Starling-like control of rotary blood pumps: an in-vitro evaluation, PLoS ONE. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5315328/
- Precisión del estimador de flujo del HVAD. https://research.monash.edu/en/publications/heartware-hvad-flow-estimator-accuracy-for-left-and-right-ventric/
- Estimación de flujo sin sensores en bombas rotatorias (revisión). https://arxiv.org/pdf/1709.08219
- Simulador híbrido para succión (2023). https://www.frontiersin.org/articles/10.3389/fphys.2023.1155032/full
- Henderson et al. (2018), Deep RL that Matters. https://arxiv.org/pdf/1709.06560
- Colas, Sigaud y Oudeyer (2018), How Many Random Seeds? https://arxiv.org/abs/1806.08295
- Agarwal et al. (2021), Deep RL at the Edge of the Statistical Precipice. https://arxiv.org/abs/2108.13264
- Phan et al. (2020), Neural Simplex Architecture. https://arxiv.org/abs/1908.00528
- Kwon, Lamp y Feng (2026), Safety Generalization Under Distribution Shift in Safe RL: A Diabetes Testbed. https://arxiv.org/abs/2601.21094
- RM temporizadas (2025). arXiv 2512.17637 (citada en el HANDOFF).
- Reesink et al. (2007), Artif Organs: succión en ovejas y modelo computacional (solo se leyó un resumen vía búsqueda).
- Karantonis et al. (2007), EMBS: estimación del flujo (citada en el HANDOFF; no releída).
