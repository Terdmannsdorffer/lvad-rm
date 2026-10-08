# Fase 1: modelo y validación (objetivos contra resultados)

| ID | Prueba | Objetivo (fuente) | Resultado | Estado |
| --- | --- | --- | --- | --- |
| V1 | Elastancia, HR 60, Emax 2, Emin 0.06 | pico ≈ 2 en t ≈ Tmax = 0.35 s (Fig. 3) | pico 1.992 mmHg/ml en t = 0.349 s | cumple |
| V2 | Sano sin bomba, HR 75, Emin 0.05, Vtot = 281.5 ml | 117/77, media 99, gasto 5.21, SV 69.5 (Fig. 5) | 117.0/77.4, media 98.9 (AoP; 95.5 en x3), gasto 5.20, SV 69.3; error máx. +0.6% | cumple |
| V2 | Sano sin bomba, HR 75, Emin 0.06, Vtot = 287.3 ml | 117/77, media 99, gasto 5.21, SV 69.5 (Fig. 5) | 117.3/78.1, media 99.3 (AoP; 95.9 en x3), gasto 5.16, SV 68.8; error máx. +1.4% | cumple |
| V3 | ESPVR, poscarga Rs 0.5-2.0, EDV fijo | pendiente 1.98, intercepto 10.91 ml (Fig. 6) | pendiente 1.995, intercepto 11.56 ml (fin de sístole = cierre aórtico) | pendiente cumple; intercepto +0.65 ml |
| V3 | ESPVR, precarga Rm 0.005-0.8 | pendiente 1.92, intercepto 8.84 ml (Fig. 6) | pendiente 1.926, intercepto 9.55 ml (fin de sístole = cierre aórtico) | pendiente cumple; intercepto +0.71 ml |
| V4 | Rampa 12-18 krpm en 60 s, Rs 1 (emin0.06_hr60) | succión ≈ 15 500 rpm; la envolvente del Qp mínimo sube y cae (Fig. 12) | pico de la envolvente en 15450 rpm (-0.3%) | cumple |
| V5 | Velocidad de succión contra Rs (emin0.06_hr60) | 17 400 / 15 500 / 14 800 / 14 400 / 14 100 rpm, ±5% (Figs. 14-15) | 0.5: 17150 (-1.4%); 1.0: 15450 (-0.3%); 1.5: 14750 (-0.3%); 2.0: 14450 (+0.3%); 2.5: 14250 (+1.1%); RMS +0.8% | cumple |
| V8 | Velocidad de cierre aórtico (estado estacionario, HR 60) | nueva | Rs 0.5: 11510; Rs 1.0: 10841; Rs 1.5: 10572; Rs 2.0: 10427; Rs 2.5: 10335. Con Rs 1: Emax 0.6: 7977; Emax 0.8: 8738; Emax 1.0: 9297; Emax 1.25: 9825; Emax 1.5: 10235; Emax 2.0: 10841 rpm | calculada |
| V9 | Conservación y régimen periódico | Vtot constante; periódico en pocas decenas de latidos | deriva relativa máx. 1.6e-13 en 200 latidos; periódico (ΔEDV < 1e-3 ml) en ≤ 4 latidos | cumple |

## Otras verificaciones

- Forma de volumen contra forma de presión (Radau, rtol 1e-10): diferencia máx. de LVP 2.9e-09 mmHg.
- Bomba, forma literal del paper: L* = 0.00363, R* = -0.0353 (polo de la rama +9.7 1/s, inestable). Desde el estado periódico a 12 krpm, |Qp| > 1e4 ml/s (hacia atrás) en 0.12 s; con ω = 0 en 0.27 s. Forma consistente: polo -6.5 1/s.

## Ventana de operación (estado estacionario, Emin 0.06, HR 60)

| Condición | ω reflujo termina | ω cierre aórtico | ω succión |
| --- | --- | --- | --- |
| Emax 2, Rs 0.5 | 8594 | 11510 | 17140 |
| Emax 2, Rs 1.0 | 10438 | 10841 | 15443 |
| Emax 2, Rs 1.5 | 11302 | 10572 | 14814 |
| Emax 2, Rs 2.0 | 11834 | 10427 | 14482 |
| Emax 2, Rs 2.5 | 12194 | 10335 | 14270 |
| Rs 1, Emax 0.6 | 7778 | 7977 | 15469 |
| Rs 1, Emax 0.8 | 8525 | 8738 | 15462 |
| Rs 1, Emax 1.0 | 9066 | 9297 | 15457 |
| Rs 1, Emax 1.25 | 9566 | 9825 | 15451 |
| Rs 1, Emax 1.5 | 9936 | 10235 | 15448 |
| Rs 1, Emax 2.0 | 10438 | 10841 | 15443 |

## Paso de integración

| dt (s) | error máx. (1 latido y periódico, todos los casos) |
| --- | --- |
| 2.0e-05 | 0.003% |
| 5.0e-05 | 0.007% |
| 1.0e-04 | 0.012% |
| 1.5e-04 | 0.019% |
| 2.0e-04 | 0.038% |

Límite de estabilidad de RK4 con la válvula aórtica abierta (Emax 2): 1.92e-04 s. dt recomendado (error < 1% y dt ≤ 0.8 del límite): 0.00015 s, en la frecuencia más baja (HR 60).

## Rendimiento en GPU (float32)

Dispositivo: cuda:0; N_sub = 6720 subpasos por latido.

| Entornos | unroll | latidos/s | ms por latido del lote | compilación (s) |
| --- | --- | --- | --- | --- |
| 1024 | 1 | 5372 | 190.6 | 0.7 |
| 1024 | 4 | 7236 | 141.5 | 1.8 |
| 1024 | 8 | 6715 | 152.5 | 3.7 |
| 1024 | 16 | 6947 | 147.4 | 8.1 |
| 2048 | 1 | 10686 | 191.6 | 0.7 |
| 2048 | 4 | 14500 | 141.2 | 2.0 |
| 2048 | 8 | 13807 | 148.3 | 3.7 |
| 2048 | 16 | 14628 | 140.0 | 8.7 |
| 4096 | 1 | 21173 | 193.5 | 0.7 |
| 4096 | 4 | 23346 | 175.4 | 1.6 |
| 4096 | 8 | 25941 | 157.9 | 3.4 |
| 4096 | 16 | 26094 | 157.0 | 8.2 |
| 8192 | 1 | 35620 | 230.0 | 0.7 |
| 8192 | 4 | 37632 | 217.7 | 1.7 |
| 8192 | 8 | 42150 | 194.4 | 3.3 |
| 8192 | 16 | 39913 | 205.2 | 8.0 |
| 16384 | 1 | 47962 | 341.6 | 0.8 |
| 16384 | 4 | 50636 | 323.6 | 2.6 |
| 16384 | 8 | 40796 | 401.6 | 5.2 |
| 16384 | 16 | 42506 | 385.5 | 10.4 |
| 32768 | 1 | 68246 | 480.1 | 1.0 |
| 32768 | 4 | 63191 | 518.6 | 2.6 |
| 32768 | 8 | 55774 | 587.5 | 5.5 |
| 32768 | 16 | 57221 | 572.7 | 10.5 |

float32 contra float64 (régimen periódico, dt recomendado):

- no_pump_hr75: error máx. 0.002% (aop_min), deriva de Vtot -2.4e-05 ml
- pump_9k_avo_hr60: error máx. 0.025% (qp_min), deriva de Vtot +1.7e-02 ml
- pump_13k_hr60: error máx. 0.001% (lvp_max), deriva de Vtot -6.7e-04 ml
- pump_15.8k_suction_hr60: error máx. 0.128% (qp_min), deriva de Vtot +2.2e-02 ml
- pump_7k_backflow_rs2_hr60: error máx. 0.001% (co), deriva de Vtot +1.3e-04 ml
- pump_11k_emax0.6_hr60: error máx. 0.003% (lvp_max), deriva de Vtot -3.2e-03 ml
- pump_14k_exercise_rs0.6_hr135: error máx. 0.003% (lvp_max), deriva de Vtot +1.8e-03 ml
