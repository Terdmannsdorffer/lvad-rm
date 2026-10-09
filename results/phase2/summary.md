# Fase 2: paciente, escenarios, entorno y líneas base

## Paciente con insuficiencia bajo LVAD (Fernández-Zapico 2024, Tabla 1)

| Variable | Objetivo | Rs libre | Rs fijo 0.9 |
| --- | --- | --- | --- |
| co | 4.9–5.1 | 5.01 | 4.71 ✗ |
| qlv | 0.7–1.2 | 0.95 | 0.95 |
| lap_mean | 14.0–16.0 | 14.98 | 14.29 |
| aop_mean | 82.0–86.0 | 84.45 | 88.11 ✗ |
| aop_max | 98.0–100.0 | 94.40 ✗ | 98.12 |
| aop_min | 68.0–75.0 | 73.08 | 76.13 ✗ |
| lvv_min | 129.0–131.0 | 130.45 | 130.74 |
| edv | 179.0–183.0 | 180.13 | 179.54 |

rs_free: Emax 1.231, Emin 0.1218, V0 73.2 ml, Vtot 345.4 ml, ω 8240 rpm, Rs 0.793; costo 3.63e-04; 7/8 en rango.

rs_fixed_0.9: Emax 1.955, Emin 0.1588, V0 100.0 ml, Vtot 347.4 ml, ω 8325 rpm, Rs 0.900; costo 3.71e-03; 5/8 en rango.

Elegido: **rs_free** (✗ = fuera de rango).

## Ventana de operación del paciente (estado estacionario, velocidad constante)

| Variación | ω reflujo > 1 ml | ω reflujo > 0 | ω cierre aórtico | ω succión | reflujo en ω cierre (ml/latido) |
| --- | --- | --- | --- | --- | --- |
| rs = 0.5 | 6790 | 7096 | 9391 | — | 0.0 |
| rs = 0.8 | 7842 | 8137 | 8994 | — | 0.0 |
| rs = 1.0 | 8302 | 8594 | 8841 | 15594 | 0.0 |
| rs = 1.5 | 9159 | 9568 | 8617 | 14961 | 3.3 |
| rs = 2.0 | 9878 | 10279 | 8495 | 14630 | 6.8 |
| emax_scale = 0.5 | 6126 | 6435 | 7115 | — | 0.0 |
| emax_scale = 0.75 | 7169 | 7467 | 8261 | — | 0.0 |
| emax_scale = 1.0 | 7823 | 8118 | 9000 | — | 0.0 |
| emax_scale = 1.25 | 8280 | 8573 | 9533 | — | 0.0 |
| emax_scale = 1.5 | 8617 | 8909 | 9945 | — | 0.0 |
| vtot_scale = 0.85 | 6996 | 7305 | 8091 | 14357 | 0.0 |
| vtot_scale = 0.9 | 7282 | 7586 | 8405 | 14949 | 0.0 |
| vtot_scale = 1.0 | 7823 | 8118 | 9000 | — | 0.0 |
| vtot_scale = 1.1 | 8330 | 8618 | 9559 | — | 0.0 |
| hr = 60.0 | 7815 | 8122 | 9004 | — | 0.0 |
| hr = 80.0 | 7823 | 8118 | 9000 | — | 0.0 |
| hr = 100.0 | 7668 | 8072 | 8855 | — | 0.0 |
| hr = 135.0 | 6992 | 7469 | 8472 | — | 0.0 |

— en ω succión: no hay succión bajo 16 krpm. — en reflujo en ω cierre: no hay reflujo en esa velocidad.

## Controlador de Simaan (V6, V6b, V6c)

- V6, publicado (c = 0.76, K = 152): diverge; velocidad máx. 18000 rpm, 18 latidos con succión de 60.
- V6, c = 0.15 (K = 30): 14735 rpm a los 10 s y 15258 a los 20 s (Fig. 16: ~15 000 y ~15 250); final 15443; 10 latidos con succión.
- V6b, c = 0.15 sin límite: colapsa a 6070 rpm tras la rampa de Rs. Con límite de 250 rpm/s: 15482 → 14470 rpm (Fig. 17: 15 250 → 14 250), 29 latidos con succión.

| Rs | min-flow: ω, gasto, PAM | Tabla V | Δp: ω, gasto, PAM | Tabla V |
| --- | --- | --- | --- | --- |
| 0.5 | 17147, 20.36, 188 | 14.4, 136 | 11404, 10.69, 107 | 11.7, 110 |
| 1.0 | 15437, 10.48, 185 | 7.5, 133 | 10077, 5.95, 112 | 6.5, 120 |
| 2.0 | 14567, 5.32, 183 | 3.8, 132 | 9173, 3.16, 115 | 3.7, 137 |

## Dificultad: 2048 escenarios aleatorios de 300 latidos, rangos por defecto (relativos al paciente)

| Política | Episodios con violación N=3 / 5 / 10 | Violaciones / 100 latidos (N=5) | AVO | SUC | BACK | LOW | Gasto medio | Retorno (N=5) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const_pop | 40% / 40% / 40% | 6.1 | 69% | 0.0% | 21% | 36% | 5.15 | -56 ± 50 |
| const_clinician | 32% / 31% / 31% | 2.9 | 85% | 0.0% | 11% | 36% | 5.31 | -33 ± 41 |
| simaan_minflow | 58% / 58% / 58% | 7.6 | 62% | 0.2% | 25% | 36% | 5.36 | -69 ± 49 |
| random | 89% / 86% / 82% | 7.7 | 59% | 1.0% | 33% | 33% | 5.78 | -90 ± 42 |

## Dificultad: 2048 escenarios aleatorios de 300 latidos, rangos amplios (Rs 0.5–2.0 absoluto)

| Política | Episodios con violación N=3 / 5 / 10 | Violaciones / 100 latidos (N=5) | AVO | SUC | BACK | LOW | Gasto medio | Retorno (N=5) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const_pop | 52% / 52% / 52% | 8.2 | 59% | 0.0% | 49% | 72% | 3.79 | -107 ± 64 |
| const_clinician | 33% / 32% / 32% | 3.0 | 85% | 0.0% | 54% | 71% | 3.78 | -99 ± 68 |
| simaan_minflow | 65% / 65% / 65% | 9.3 | 53% | 0.2% | 48% | 71% | 3.94 | -111 ± 60 |
| random | 92% / 90% / 86% | 8.6 | 54% | 2.1% | 49% | 63% | 4.24 | -123 ± 50 |

## Protocolos de evaluación (1206 latidos): violaciones / 100 latidos (N = 5)

| Protocolo | const_pop | const_clinician | simaan_minflow |
| --- | --- | --- | --- |
| ferreira_exercise | 0.0 (gasto 6.6) | 0.1 (gasto 6.8) | 15.2 (gasto 8.7) |
| ferreira_hypertension | 0.0 (gasto 3.6) | 15.1 (gasto 3.8) | 0.0 (gasto 3.1) |
| fz_sequential | 1.8 (gasto 5.0) | 5.1 (gasto 5.3) | 6.6 (gasto 5.4) |
