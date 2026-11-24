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
