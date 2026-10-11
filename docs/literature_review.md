# Revisión bibliográfica: estado del arte y posición del proyecto

Búsqueda hecha el 10 de octubre de 2026 (tarea abierta del HANDOFF, §2).

## Protocolo

- **Fuente:** búsqueda web (13 consultas, modo extendido en las críticas). No hubo acceso directo a PubMed, IEEE Xplore ni Scopus. **Solo se leyeron resúmenes y fragmentos, no textos completos.** Las cifras de cada trabajo deben verificarse en la fuente antes de citarlas.
- **Ventana:** 2024–2026, más los trabajos anteriores que son referencia obligada.
- **Ejes de búsqueda:**
  - (A) RL para control de VAD;
  - (B) controladores fisiológicos con apertura aórtica o sin sensores;
  - (C) detección de la apertura aórtica desde señales de la bomba;
  - (D) detección de succión;
  - (E) reward machines y lógica temporal, en medicina y con ruido;
  - (F) revisiones.
- **Consultas** (resumidas):
  - "deep reinforcement learning ventricular assist device controller 2024–2026";
  - "reward machine OR temporal logic reinforcement learning medical device";
  - citas y seguimientos de "Reward Machines for Deep RL in Noisy and Uncertain Environments";
  - "LVAD physiological control aortic valve opening controller";
  - "signal temporal logic rotary blood pump";
  - "Computing in Cardiology LVAD";
  - "aortic valve opening detection pump signals";
  - "suction detection deep learning";
  - "sensorless physiological control LVAD clinical".

## A. RL para control de VAD

| Trabajo | Método y sensores | Objetivo | Validación |
| --- | --- | --- | --- |
| Li T. et al. 2022, *Artif Intell Med* 128:102308 | SAC; modelo cardiovascular simple | Seguir un flujo objetivo tipo Starling | In silico, contra un PID |
| Fernández-Zapico et al. 2024, *Artif Organs* 48:1418–1429 | PPO; presiones del VI y aórtica (EDP, LVP máx., AoP máx.); recompensa que depende de la acción | Evitar succión y permitir apertura aórtica | In silico, simulador de alta fidelidad; contra velocidad constante |
| Shi et al. 2025, *Comput Methods Programs Biomed* 270:108975 | DDPG modificado con LSTM; presión y flujo aórticos | Recuperación hemodinámica y pulsatilidad en VAD percutáneo | In silico e **in vitro**; robusto a ruido de sensor |

La revisión de 2026 (arXiv 2609.20518) describe el RL para LVAD como prometedor, pero señala que necesita muchos datos, cuesta mucho cómputo y es poco interpretable. **No se encontraron estudios in vivo ni clínicos de controladores de VAD basados en RL.**

## B. Controladores fisiológicos (no RL): apertura aórtica o sin sensores

| Trabajo | Sensores | Lo relevante | Validación |
| --- | --- | --- | --- |
| Petrou et al. (ETH Zúrich), *Ann Biomed Eng* | Presión de entrada de la bomba | Multiobjetivo: adecuar el flujo a la demanda, sostener la apertura aórtica por un tiempo fijado y aumentar la pulsatilidad, con capas de protección contra succión, sobrecarga y reflujo. Clasifica el estado de la válvula con 86% de exactitud; presión de pulso ~3 veces mayor que con velocidad constante | Circulación simulada híbrida (in vitro) |
| Maw et al. 2022, *Front Cardiovasc Med* (NCT04786236) | **Sin sensores** (HVAD) | Módulos de frecuencia cardíaca, pulsatilidad, reacción a succión y supervisor | **Piloto clínico**: 6 pacientes y 13 pruebas, sin intervenciones del sistema de seguridad |
| Santos y Cestari 2026, arXiv 2609.24966 | Presión en la cánula de entrada | Estima HR y presión de llenado; eliminó succión y reflujo en los 12 de 27 escenarios simulados en que aparecían | In silico |
| Perfiles óptimos de velocidad (sistema conmutado), *Ann Biomed Eng* 2021 | Basado en modelo | Perfiles constantes por tramos que se parecen a contrapulsación | In silico, un paciente |
| Algoritmo basado en trabajo sistólico, *ASAIO J* | — | Control según el trabajo sistólico ventricular | In vivo |
| Modulación intermitente comercial | — | Algunos fabricantes bajan la velocidad periódicamente para favorecer la apertura **sin conocer el estado de la válvula** (RWTH Aachen). Los ejemplos que conocemos son el ciclo Lavare del HVAD y el Artificial Pulse del HeartMate 3; esto no se verificó en esta búsqueda | Clínico (dispositivos comerciales) |
| Estudio cruzado con ejercicio, *Heart* 2022 (NCT05063006) | Ecocardiografía | Ajuste continuo de la velocidad para preservar la apertura aórtica periódica durante el ejercicio | Clínico |

## C. Detección de la apertura aórtica desde señales de la bomba

- **Hayward et al. 2015 y Clifford et al. 2019** (pacientes con HVAD): usan el área bajo la densidad espectral de potencia de la velocidad de la bomba. Clifford reporta AUC 0.95, sensibilidad 86% y especificidad 93%, con precisión similar en ritmo sinusal, fibrilación auricular y extrasístoles; 18 pacientes, con ecocardiografía como referencia.
- **Patentes** que generan un indicador de apertura a partir de la velocidad de la bomba.
- **CinC 2026:** estudio de factibilidad con fonocardiografía.
- **No se encontraron modelos de deep learning para detectar la apertura aórtica** desde señales de bomba.

## D. Detección de succión desde señales de la bomba

- Área madura. Karantonis et al. (CART sobre la velocidad) reportan 99.2% de sensibilidad y 98.3% de especificidad.
- También hay SVM (Wang y Simaan), redes neuronales (>98%), CART con 4 rasgos del flujo (*Front Med Technol* 2026) y un simulador híbrido para estudiar succión (*Front Physiol* 2023).

## E. Reward machines y lógica temporal

- **Fundamentos:** Toro Icarte et al., ICML 2018 y JAIR 2022.
- **Con ruido o incertidumbre:**
  - Li et al. 2022, arXiv 2211.10902: aprender directamente la creencia sobre estados de la RM con una red recurrente supervisada; es el antecedente de TDM.
  - **Li et al. 2024, NeurIPS** (Naive, IBU, TDM; dominios de juguete y robótica): IBU diverge con ruido correlacionado y TDM se acerca al oráculo.
  - PROB-IRM, arXiv 2408.14871: aprende la RM con etiquetas ruidosas usando programación lógica inductiva; dominios de grilla.
  - Neural reward machines totalmente aprendibles, arXiv 2509.19017.
  - **No se encontraron aplicaciones de reward machines con ruido a control fisiológico.**
- **En dispositivos médicos:**
  - marcapasos, con Dole et al. (AAAI 2023, citado en el HANDOFF) y Komp et al. 2024 (arXiv 2411.01750, aprenden la RM desde demostraciones y validan contra una especificación de Boston Scientific);
  - en insulina y ventilación mecánica, las recompensas son diseñadas a mano o basadas en guías clínicas, sin reward machines.
- **Extensiones temporales:** reward machines temporizadas (arXiv 2512.17637, en el HANDOFF) y reward machines para lógica temporal de señales (arXiv 2608.13625; solo se vio su lista de referencias).
- **Lógica temporal en bombas de sangre:** no se encontró ninguna especificación formal (STL u otra) aplicada al control de velocidad de LVAD.

## Posición del proyecto

**Novedad** (dentro de los límites de esta búsqueda):

1. Es la primera combinación de reward machines (o cualquier especificación temporal formal) con control de LVAD.
2. Es el primer controlador de LVAD basado en RL que usa solo señales de la bomba, con la restricción temporal de apertura aórtica explícita y detección ruidosa del evento.
3. Es la primera aplicación de reward machines con ruido (Naive, IBU, TDM) a control fisiológico, con detectores cuyos errores dependen del estado fisiológico y están correlacionados en el tiempo.

**Trabajos más cercanos:**

- **Fernández-Zapico 2024:** RL con objetivo de apertura aórtica, pero usa presiones y no tiene estructura temporal.
- **Petrou:** clásico, con objetivo de tiempo de apertura, presión de entrada y validación in vitro.
- **Maw 2022:** sin sensores y con piloto clínico, pero sin objetivo explícito de apertura.
- **Santos 2026:** presión de entrada, in silico.

**Hallazgos nuestros que aportan a la literatura de reward machines con ruido:**

- En control, una creencia mala se paga con conservadurismo más que con violaciones. Los colapsos del entrenamiento que vimos venían del recorte del valor en PPO, no de la creencia (`phase6_decisions.md`, sección G).
- TDM corrige fallas del detector que dependen del régimen si sus datos cubren ese régimen.
- Contra una regla de modulación con el mismo detector en el lazo, TDM tiene ~4 veces menos violaciones.

**Debilidades frente al estado del arte** (para declarar en el paper):

- Modelo 0D sin ventrículo derecho ni barorreflejo; Fernández-Zapico usa un simulador de alta fidelidad.
- Sin validación in vitro (Shi 2025 y Petrou sí la tienen).
- Bomba axial tipo HeartMate II; los dispositivos actuales son centrífugos.
- El detector MLP es optimista frente a la detección clínica (AUC 0.997 contra 0.95). Por eso conviene un escenario principal con error clínico.

**Citas mínimas para el paper:** Simaan 2009, Ferreira 2009, Fernández-Zapico 2024, Li T. 2022, Shi 2025, Petrou, Maw 2022, Clifford 2019, Hayward 2015, Toro Icarte 2018 y 2022, Li et al. 2022 y 2024, Komp 2024, Dole 2023, y la revisión de VAD de 2026. Opcionales: Mahr 2017 (apertura aórtica intermitente y trombosis) y Karantonis 2007.

## Pendiente para completar la revisión

- **"Citado por" en Semantic Scholar o Google Scholar** de Li et al. 2024 (arXiv 2406.00120), Fernández-Zapico 2024 (doi:10.1111/aor.14845) y Shi et al. 2025 (doi:10.1016/j.cmpb.2025.108975).
- **PubMed:** ("left ventricular assist device" OR LVAD) AND ("reinforcement learning" OR "machine learning") AND (control OR speed), de 2024 en adelante. También ("aortic valve opening") AND (LVAD) AND (detection OR control).
- **Actas de CinC 2024–2026:** buscar "ventricular assist" (los DOI tienen la forma 10.22489/CinC.2025.xxx).

## Fuentes consultadas

- Fernández-Zapico et al. 2024: https://onlinelibrary.wiley.com/doi/10.1111/aor.14845 · https://pubmed.ncbi.nlm.nih.gov/39289857/
- Li T. et al. 2022: https://www.sciencedirect.com/science/article/abs/pii/S0933365722000732
- Shi et al. 2025 (pVAD): https://pubmed.ncbi.nlm.nih.gov/40712369/ · https://www.sciencedirect.com/science/article/abs/pii/S016926072500392X
- Revisión de modelado de VAD 2026: https://arxiv.org/abs/2609.20518
- Santos y Cestari 2026: https://arxiv.org/pdf/2609.24966
- Petrou et al.: https://link.springer.com/article/10.1007/s10439-017-1919-0
- Maw et al. 2022: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9081924/
- Perfiles óptimos 2021: https://link.springer.com/article/10.1007/s10439-021-02848-2
- RWTH Aachen: https://publications.rwth-aachen.de/record/786644/files/786644.pdf
- Heart 2022: https://heart.bmj.com/content/108/13/1055
- Algoritmo por trabajo sistólico (ASAIO): https://www.ovid.com/jnls/asaiojournal/abstract/10.1097/mat.0000000000001722
- Detección de apertura aórtica en HVAD (Victor Chang): https://eprints.victorchang.edu.au/770
- CinC 2026, fonocardiografía: https://cinc.org/2026/Program/accepted/297.html
- Detección de succión 2026: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12835374/
- Karantonis et al.: https://research.monash.edu/en/publications/automated-non-invasive-detection-of-pumping-states-in-an-implanta/
- Li et al. 2024 (NeurIPS): https://arxiv.org/abs/2406.00120 · código: https://github.com/andrewli77/reward-machines-noisy-environments
- Li et al. 2022 (abstracciones simbólicas ruidosas): https://arxiv.org/pdf/2211.10902
- PROB-IRM: https://arxiv.org/pdf/2408.14871
- Neural reward machines totalmente aprendibles: https://arxiv.org/pdf/2509.19017
- Komp et al. 2024: https://arxiv.org/abs/2411.01750
- Reward machines para STL 2026: https://arxiv.org/pdf/2608.13625
- Simulador híbrido para succión 2023: https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1155032/pdf
