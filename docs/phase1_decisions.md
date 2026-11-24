# Fase 1: decisiones, supuestos y discrepancias

Complementa [results/phase1/summary.md](../results/phase1/summary.md).

## Decisiones tomadas

| Tema | Decisión | Justificación |
| --- | --- | --- |
| Forma del modelo | Forma de volumen (estado LVV) | Coincide con la forma de presión hasta 3e-9 mmHg (Radau, rtol 1e-10) y evita los picos de Ċ/C |
| Signo de la bomba | (Li + Lo − β1) dQp/dt = x1 − x4 − (Ri + Ro + Rk − β0) Qp + β2 ω² | La forma literal (ecs. 11–13) tiene R* = −0.0353 (polo +9.7 1/s) y bombea hacia atrás: |Qp| > 1e4 ml/s en 0.12 s |
| HR de las simulaciones con bomba | 60 lpm | Fig. 16(b): 30 latidos en 30 s. La ventana de 2.5 s con 3 mínimos también calza con un latido por segundo |
| Corazón de las simulaciones con bomba | Emax 2, Emin 0.06, V0 10, Vtot = 287.3 ml | Con su Vtot calibrado en la Fig. 5, Emin 0.06 reproduce la Fig. 15 con un RMS de 0.8% (Emin 0.05: 2.3%) |
| Calibración de Vtot | Mínimos cuadrados (errores relativos) sobre PAS, PAD, PAM, gasto y SV de la Fig. 5 | Emin 0.05 → 281.5 ml (error máx. 0.6%); Emin 0.06 → 287.3 ml (error máx. 1.4%) |
| Nodo de la presión media | x4 (AoP) | La Fig. 5 grafica AoP; con x4 la media da 98.9–99.3 mmHg (en x3 da 95.5–95.9) |
| Fin de sístole (ESPVR) | Instante de cierre de la válvula aórtica | Es la esquina del lazo PV. En t = Tmax la recta es exacta por construcción (pendiente 1.992, intercepto 10) |
| V3 poscarga | Vtot reajustado para cada Rs, de modo que el EDV quede fijo (135.9 ml) | La Fig. 6(a) mantiene el EDV constante |
| Velocidad de succión | Velocidad media del latido en que la envolvente del Qp mínimo alcanza su máximo (resolución: 1 latido = 100 rpm) | Criterio de Simaan |
| dt | 1.5e-4 s a HR 60, con N_sub = 6720 = 64 × 105 subpasos por latido | Error < 0.02% en todos los casos; límite de estabilidad de RK4 con la válvula aórtica abierta: 1.92e-4 s |
| Precisión | float64 en CPU para validar; float32 en GPU | Error máx. de float32 de 0.13% (Qp mínimo en succión) |

## Discrepancias con los papers

1. **Simaan, Tabla V contra Figs. 16–17.** Con el modelo corregido, la velocidad de succión en régimen periódico con Rs = 1 da un gasto de ≈10.3 l/min (Qp mínimo ≈ 168 ml/s). Esto coincide con la Fig. 16(b) (Qp ≈ 165–177 ml/s), pero no con los 7.5 l/min de la Tabla V. Se priorizan las figuras (decidido el 8 de octubre de 2026). V6c se reportará como discrepancia en la Fase 2.
2. **Interceptos de la ESPVR (V3):** salen 11.56 y 9.55 ml contra 10.91 y 8.84; las pendientes coinciden (1.995 contra 1.98 y 1.926 contra 1.92). El intercepto depende de cómo se define el fin de sístole: el paper no lo dice, y probablemente marcó los puntos a ojo.
3. **Ferreira 2009, funciones de pertenencia:** NB y PB son hombros (saturan más allá de ±2), no triángulos. Los centros de salida son −0.05, −0.025, 0, 0.025 y 0.05.

## Hallazgos que afectan el diseño de la RM (Fase 2)

- **SUC:** la resistencia de succión se activa ≈100 rpm por encima del pico de la envolvente de Simaan. La etiqueta SUC (x1 ≤ 1 mmHg) y la velocidad de succión del paper prácticamente coinciden.
- **AVO contra BACK:** con Vtot fijo y HR 60, la válvula aórtica deja de abrir en 10.3–11.5 krpm (según Rs) y en 8.0–10.8 krpm (según Emax). El reflujo instantáneo por la bomba (Qp mínimo < 0) termina en 7.8–12.2 krpm. Con Rs ≥ 1.5 **no hay velocidad constante que dé apertura aórtica sin reflujo instantáneo**, y con Rs 1 la ventana mide solo ~200–400 rpm. Mantener la apertura sin reflujo exige modular la velocidad latido a latido (justo lo que la RM puede premiar), o bien relajar la definición de BACK (por ejemplo, flujo medio de la bomba < 0 o volumen regurgitado > umbral). **Decisión pendiente.**
- **Deriva de Vtot en float32:** hasta ≈0.02 ml cada 30 latidos (unos 0.2 ml en un episodio de 300 latidos). Propuesta para el entorno: proyectar de vuelta al Vtot objetivo al final de cada latido (corrigiendo x2) y llevar Vtot como variable explícita, que de todos modos hace falta para las caídas de precarga.

## Supuestos (no están en los papers)

- ε_AVO = 0.5 ml por latido.
- Vtot fijo al variar Emax en V8 (sin barorreflejo ni retención de líquido).
- Las condiciones iniciales no importan: el régimen es periódico a precisión de máquina en unos 10 latidos.
