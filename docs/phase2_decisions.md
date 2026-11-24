# Fase 2: decisiones, supuestos y discrepancias

Complementa [results/phase2/summary.md](../results/phase2/summary.md).

## Decisiones tomadas

| Tema | Decisión | Justificación |
| --- | --- | --- |
| Paciente con insuficiencia (§7) | Ajuste por mínimos cuadrados de Emax, Emin, V0, Vtot, ω **y Rs** contra la Tabla 1 de Fernández-Zapico. Resultado: Emax 1.231, Emin 0.1218, V0 73.2 ml, Vtot 345.4 ml, Rs 0.793, ω 8 240 rpm, HR 80 | 7 de 8 objetivos en rango (falta la AoP máxima: 94.4 contra 98–100). Con Rs fijo en 0.9 el costo es 10 veces mayor, quedan 5 de 8 en rango, V0 choca con su cota y hay reflujo en el punto base. Nuestro Rs retorna a la aurícula, no a la presión venosa central (CVP), así que no es exactamente su SVR |
| Variables de Fernández-Zapico | TCO → gasto; QLV → flujo aórtico medio; enclavamiento → LAP media; AOP → AoP media; AOPES/AOPED → AoP máx./mín.; LVES → LVV mín.; LVED → LVV al final del latido | Definiciones más cercanas disponibles en un modelo 0D |
| BACK | Volumen regurgitado por la bomba > 1 ml por latido (≈0.08 l/min a HR 80) | Con la definición estricta (Qp < 0 en algún instante) cuentan regurgitaciones mínimas, sin relevancia clínica. Umbral configurable (`EnvConfig.back_ml`) |
| LOW | Gasto < 4.4 l/min | Índice cardíaco 2.2 l/min/m² con 2 m² de superficie corporal (supuesto). El objetivo del término de perfusión sigue siendo 5.0 l/min (Fernández-Zapico) |
| AVO | Volumen eyectado por la válvula aórtica > 0.5 ml | Igual que en la Fase 1 |
| Rangos de escenarios | Multiplicadores del paciente según el "rango máximo" de Fernández-Zapico: Rs × 0.44–1.56, Emax × 0.5–1.5, Vtot × 0.85–1.10; HR 60–135 (60–100 al inicio) | El rango absoluto del HANDOFF (Rs 0.5–2.0) se pensó para el corazón de Simaan (Rs 1.0); para este paciente (Rs 0.79) llega a 2.5 veces el valor base y genera 72% de latidos con gasto bajo. Se conserva como `WIDE_SCENARIOS` para generalización |
| Eventos | Hasta 3 por episodio (parámetro, inicio, valor y número al azar), rampas lineales de 10 s para HR y 50 s para el resto, aplicadas al inicio de cada latido | Como Fernández-Zapico (según Ursino 1998) |
| Precarga | Fuente de volumen q_vol en el nodo auricular, calculada en cada latido para llegar al Vtot programado | Es el mecanismo de §8; además corrige la deriva de Vtot en float32 (test: Vtot sigue el programa con error relativo < 1e-4) |
| Episodios | Largo fijo (300 latidos), 15 latidos de calentamiento en el reset, sin auto-reset por paso | Los entornos vectorizados quedan sincronizados y se reinician juntos en el entrenamiento; un auto-reset por paso obligaría a recalcular el calentamiento en cada paso |
| Observación de bomba | 7 estadísticos de Qp (media, mín., máx., pulsatilidad, desviación estándar, dQ/dt máx. y mín.) calculados sobre 64 muestras con ruido, más ω, HR (de la duración del latido) y las últimas 4 acciones | La HR se mide en la señal de la bomba. Ruido uniforme con SNR de 25 dB (supuesto; Ferreira dice que los sensores reales superan los 20 dB) |
| Observación con presiones | (EDP − 14, LVPmáx − AoPmáx), ambos /10, más sus valores del latido anterior | MDP de Fernández-Zapico |
| Velocidad del clínico | 0.97 × la mayor velocidad que abre la válvula aórtica en el estado inicial del episodio | Margen de 3% (supuesto) |
| Controlador de Simaan (línea base) | c = 0.15 rpm/(ml/s) por actualización (K = 30), límite de 250 rpm/s, versión por latido con escalones de 0.005 s | Ver discrepancias |
| Controlador Δp | Versión por latido de Giridharan y Skliar (ganancia 0.004 rad/s/mmHg cada 0.005 s, referencia 75 mmHg) | Para V6c |
| Protocolos de evaluación | Ferreira: ejercicio (Rs × 0.8 y HR 90 a los 200 s; Rs × 0.6 y HR 135 a los 400 s) e hipertensión (Rs × 1.8 a los 200 s). Secuencial tipo Fernández-Zapico: Vtot, HR, Rs y Emax suben y bajan de a uno, con 20 s de estabilización (840 s, 1 206 latidos) | Valores de Rs relativos al paciente |

## Discrepancias con los papers

1. **Controlador de Simaan, ganancia.** Con c = 0.76 aplicado cada 0.005 s, K = c/Δt = 152 rpm por ml/s. Como el Qp mínimo sube ≈0.031 ml/s por rpm, el lazo (que es de retroalimentación positiva) tiene ganancia ≈4.7 y se dispara. La Fig. 16 implica K ≈ 29 (1 650 rpm para una subida de 57 ml/s del mínimo), lo que equivale a actualizar cada ~0.026 s. Con K = 30 se reproduce la subida de la Fig. 16 (14 735 rpm a los 10 s y 15 258 a los 20 s, contra ~15 000 y ~15 250), pero el controlador queda en el borde de succión.
2. **V6b no se reproduce.** Sin límite de tasa, el controlador colapsa al entrar en succión tras la rampa de Rs. Con 250 rpm/s sigue la nueva velocidad de succión (14 470 rpm contra 14 250), pero con 29 latidos en succión. El paper no publica estos detalles (estimador, límites ni condición inicial).
3. **V6c.** En el borde de succión, el min-flow da gasto 20.4 / 10.5 / 5.3 l/min y PAM ≈185 mmHg. Esto es consistente con las Figs. 14–16 y no con la Tabla V, lo que confirma la discrepancia de la Fase 1. El controlador Δp sí se acerca a la Tabla V (10.7 / 5.95 / 3.16 l/min contra 11.7 / 6.5 / 3.7).
4. **Presión de pulso del paciente:** la AoP máxima queda en 94 contra 98–100 mmHg. Probablemente el Windkessel de Simaan, sin pulmón ni ventrículo derecho, no la reproduce.

## Hallazgos para la Fase 3

- **La succión casi no está al alcance.** Con este paciente (V0 = 73 ml), la válvula aórtica deja de abrir en 6.3–9.9 krpm, pero la succión aparece recién sobre 10.9 krpm, incluso con Vtot × 0.6. Una política que respeta la apertura aórtica nunca llega a succión; con velocidad constante, SUC ocurre en 0% de los latidos. **Decisión (usuario, checkpoint 2): opción (a).** SUC queda en la RM como evento de seguridad poco frecuente (penalización r_suc) y el estudio de detectores ruidosos (H2) se centra en AVO. Las alternativas descartadas eran forzar la succión con escenarios más duros o un objetivo de gasto mayor, y recalibrar con V0 más chico.
- **AVO y BACK compiten cuando Rs es alto.** Con Rs ≥ ~1.3 (en escala absoluta) no hay velocidad constante que dé apertura sin regurgitar más de 1 ml. Hace falta modular la velocidad o aceptar BACK.
- **N casi no influye en las líneas base de velocidad constante.** Cuando un evento cierra la válvula, queda cerrada por tramos largos.
