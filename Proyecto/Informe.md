# Sistema BCI para Clasificación de Imaginación Motora mediante EEG y Machine Learning

**Proyecto de Interfaz Cerebro-Computadora**  
*Pontificia Universidad Católica del Perú - Curso de Introducción a Señales Biomédicas*

---

## Tabla de Contenidos

1. [Resumen](#resumen)
2. [Palabras Clave](#palabras-clave)
3. [Introducción](#introducción)
4. [Planteamiento del Problema](#planteamiento-del-problema)
5. [Propuesta de Solución](#propuesta-de-solución)
   - 5.1 [Arquitectura del Sistema](#arquitectura-del-sistema)
   - 5.2 [Preprocesamiento de Señales](#preprocesamiento-de-señales)
   - 5.3 [Extracción de Características](#extracción-de-características)
   - 5.4 [Modelos de Clasificación](#modelos-de-clasificación)
   - 5.5 [Estrategia de Validación](#estrategia-de-validación)
6. [Resultados](#resultados)
   - 6.1 [Preprocesamiento](#resultados-preprocesamiento)
   - 6.2 [Modelos Binarios](#modelos-binarios)
   - 6.3 [Modelo Multi-Clase](#modelo-multi-clase)
   - 6.4 [Comparación de Enfoques](#comparación-de-enfoques)
7. [Conclusiones](#conclusiones)
8. [Referencias](#referencias)
9. [Biografías de Autores](#biografías-de-autores)

---

## Resumen

Las interfaces cerebro-computadora (BCI) representan una tecnología prometedora para la comunicación y control de dispositivos mediante señales cerebrales, con aplicaciones en rehabilitación neurológica, asistencia a personas con discapacidad motora y neurociencia cognitiva. Este proyecto desarrolla un sistema BCI capaz de clasificar imaginación motora de manos y pies a partir de señales electroencefalográficas (EEG), utilizando técnicas de Machine Learning tradicional. Se implementó un pipeline automático completo que procesa datos del PhysioNet EEG Motor Movement/Imagery Database (109 sujetos, ~9,700 epochs), aplicando filtrado de frecuencias, remoción de artefactos mediante ICA, extracción de características CSP (Common Spatial Patterns) y Band Power, y entrenamiento de modelos LDA, SVM y Random Forest con validación cruzada LORO (Leave-One-Run-Out). Se evaluaron dos enfoques: (1) dos modelos binarios especializados para clasificar manos izquierda vs derecha y ambas manos vs ambos pies, alcanzando 62.7% de accuracy con SVM-RBF; y (2) un modelo unificado de 4 clases, alcanzando 38.9% de accuracy con LDA. Los resultados demuestran que los modelos binarios ofrecen mayor precisión para tareas específicas, mientras que el enfoque multi-clase proporciona una solución más general. Este trabajo establece una base sólida para futuros desarrollos con Deep Learning y aplicaciones en tiempo real.

---

## Palabras Clave

Interfaz Cerebro-Computadora (BCI), Electroencefalografía (EEG), Imaginación Motora, Machine Learning, Common Spatial Patterns (CSP), Support Vector Machine (SVM), Procesamiento de Señales Biomédicas, Clasificación Multi-Clase, PhysioNet Database

---

## Introducción

### Contexto y Motivación

Las interfaces cerebro-computadora (Brain-Computer Interface, BCI) son sistemas que permiten la comunicación directa entre el cerebro humano y dispositivos externos, sin requerir actividad muscular [1]. Estas tecnologías han revolucionado campos como la neurorrehabilitación, permitiendo a personas con discapacidades motoras severas controlar prótesis, sillas de ruedas o interfaces computacionales mediante la actividad cerebral [2].

La imaginación motora (motor imagery) es el paradigma BCI más estudiado, basándose en la modulación de ritmos sensoriomotores durante la simulación mental de movimientos sin ejecución física [3]. Cuando una persona imagina mover una mano o pie, regiones específicas de la corteza motora se activan, produciendo patrones característicos en las señales EEG que pueden ser detectados y clasificados mediante algoritmos de Machine Learning [4].

### Estado del Arte

Los sistemas BCI basados en imaginación motora han evolucionado significativamente en las últimas décadas. Los métodos tradicionales utilizan extracción de características mediante CSP (Common Spatial Patterns) [5] y clasificadores lineales como LDA (Linear Discriminant Analysis) [6], alcanzando accuracies de 60-70% en clasificación binaria. Recientemente, modelos de Deep Learning como EEGNet [7] y Deep ConvNet [8] han superado el 75% de accuracy en benchmarks estándar.

Sin embargo, persisten desafíos importantes: (1) alta variabilidad inter-sujeto que dificulta la generalización, (2) necesidad de calibración individual extensiva, (3) bajo desempeño en clasificación multi-clase (>2 clases), y (4) complejidad computacional que limita aplicaciones en tiempo real [9].

### Objetivos del Proyecto

Este proyecto tiene como **objetivo general** diseñar e implementar un sistema BCI capaz de clasificar imaginación motora de cuatro movimientos (mano izquierda, mano derecha, ambas manos, ambos pies) a partir de señales EEG, utilizando técnicas de Machine Learning tradicional.

Los **objetivos específicos** son:

1. Desarrollar un pipeline automático de preprocesamiento de señales EEG (filtrado, remoción de artefactos, segmentación)
2. Implementar extracción de características CSP y espectrales (Band Power)
3. Entrenar y evaluar múltiples modelos de clasificación (LDA, SVM, Random Forest)
4. Comparar dos enfoques: modelos binarios especializados vs modelo unificado multi-clase
5. Validar el sistema mediante Leave-One-Run-Out cross-validation
6. Analizar el trade-off entre precisión y generalidad en ambos enfoques

---

## Planteamiento del Problema

### Descripción del Problema

Las personas con discapacidades motoras severas (e.g., lesión medular, esclerosis lateral amiotrófica, accidente cerebrovascular) enfrentan limitaciones significativas en su capacidad de comunicación y control del entorno. Los sistemas BCI ofrecen una alternativa no invasiva que podría restaurar cierta autonomía, pero requieren clasificadores robustos capaces de decodificar intenciones motoras a partir de señales EEG ruidosas y variables.

### Desafíos Técnicos

1. **Bajo SNR (Signal-to-Noise Ratio):** Las señales EEG son altamente susceptibles a artefactos oculares, musculares y ambientales, con amplitudes en el rango de microvoltios
2. **Variabilidad Inter-Sujeto:** Diferencias anatómicas, cognitivas y de entrenamiento producen patrones EEG distintos entre individuos
3. **Variabilidad Intra-Sujeto:** Fatiga, atención y estado emocional afectan la calidad de las señales incluso en el mismo sujeto
4. **Clasificación Multi-Clase:** Distinguir >2 clases de movimientos imaginados es significativamente más complejo que clasificación binaria
5. **Limitaciones Computacionales:** Los sistemas en tiempo real requieren procesamiento de baja latencia (<300ms) para retroalimentación efectiva

### Pregunta de Investigación

**¿Es posible desarrollar un sistema BCI basado en Machine Learning tradicional que clasifique imaginación motora de manos y pies con >60% de accuracy, utilizando únicamente señales EEG no invasivas?**

Adicionalmente, se busca responder: **¿Cuál es el trade-off entre usar modelos binarios especializados versus un modelo unificado multi-clase en términos de precisión, generalidad y aplicabilidad?**

---

## Propuesta de Solución

### 5.1 Arquitectura del Sistema

El sistema propuesto sigue un pipeline modular de 5 etapas, desde la adquisición de datos crudos hasta la predicción de clases:

```
[Datos EDF] → [Preprocesamiento] → [Extracción Features] → [Clasificación] → [Predicción]
     ↓              ↓                      ↓                     ↓               ↓
  64 canales   Filtros + ICA          CSP + Band Power      LDA/SVM/RF      4 clases
  160 Hz       8-30 Hz                124-142 features      LORO CV         0,1,2,3
```

**Componentes principales:**

- **Módulo de Datos:** Carga archivos EDF y parsea anotaciones de eventos
- **Módulo de Preprocesamiento:** Filtrado, remoción de artefactos, segmentación, normalización
- **Módulo de Características:** CSP y extracción espectral (mu/beta bands)
- **Módulo de Modelos:** LDA, SVM (RBF/Linear), Random Forest
- **Módulo de Validación:** Leave-One-Run-Out cross-validation, métricas comprehensivas

### 5.2 Preprocesamiento de Señales

#### Dataset

Se utilizó el **PhysioNet EEG Motor Movement/Imagery Database** [10], que contiene:

- **109 sujetos** voluntarios sanos
- **64 canales** de EEG (sistema 10-10) a 160 Hz
- **14 runs** por sujeto con diferentes tareas motoras
- **Runs de interés:**
  - R04, R08, R12: Imaginación de mano izquierda (T1) vs derecha (T2)
  - R06, R10, R14: Imaginación de ambas manos (T1) vs ambos pies (T2)

**Fuente:** https://physionet.org/content/eegmmidb/1.0.0/

#### Filtrado de Frecuencias

Se aplicó un **filtro pasa-banda de 8-30 Hz** (método FIR) para aislar las bandas mu (8-13 Hz) y beta (13-30 Hz), asociadas a actividad sensoriomotora [11]:

```python
raw.filter(l_freq=8.0, h_freq=30.0, method='fir')
raw.notch_filter(freqs=[60.0], method='fir')  # Eliminación de ruido de línea
```

**Justificación:** La banda mu se desincroniza durante imaginación motora (Event-Related Desynchronization, ERD), mientras que beta presenta sincronización post-movimiento (Event-Related Synchronization, ERS) [12].

![Comparación de Filtrado](/Proyecto/Software/reports/figures/02_filtering_comparison.png)
*Figura 1: Efecto del filtrado pasa-banda en señales EEG. Superior: señal cruda, inferior: señal filtrada 8-30 Hz.*

#### Remoción de Artefactos

Se empleó **Independent Component Analysis (ICA)** con 15 componentes para separar y remover artefactos oculares y musculares [13]:

```python
ica = ICA(n_components=15, method='fastica', random_state=42)
ica.fit(raw)
# Remoción automática de componentes de artefacto
```

![Remoción de Artefactos](/Proyecto/Software/reports/figures/02_artifact_removal.png)
*Figura 2: Identificación y remoción de componentes de artefacto mediante ICA.*

#### Segmentación y Normalización

- **Épocas:** Ventanas de -0.5s a +4.0s relativas al evento de imaginación motora
- **Baseline:** Corrección usando 500ms pre-estímulo
- **Normalización:** Z-score por canal para estandarizar amplitudes entre sujetos

![Épocas Promediadas](/Proyecto/Software/reports/figures/02_averaged_epochs.png)
*Figura 3: Épocas promediadas por clase mostrando patrones distintivos de imaginación motora.*

### 5.3 Extracción de Características

#### Common Spatial Patterns (CSP)

CSP es una técnica de filtrado espacial que maximiza la varianza de una clase mientras minimiza la de otra, mediante la solución de un problema de valores propios generalizados [5]:

$$
\mathbf{w}^T \mathbf{\Sigma}_1 \mathbf{w} / \mathbf{w}^T \mathbf{\Sigma}_2 \mathbf{w} \rightarrow \text{max}
$$

Donde $\mathbf{\Sigma}_1$ y $\mathbf{\Sigma}_2$ son las matrices de covarianza de ambas clases.

**Implementación:**
- **6 componentes CSP** por clasificador binario (3 por clase)
- **Multi-class CSP** (One-vs-Rest) para 4 clases: 24 componentes totales

```python
csp = CSPExtractor(n_components=6)
csp.fit(X_train, y_train)
X_csp = csp.transform(X_test)  # 6 features
```

#### Band Power Features

Se calculó la potencia espectral relativa en bandas mu (8-13 Hz) y beta (13-30 Hz) para cada canal mediante transformada de Fourier:

$$
P_{rel}(f_1, f_2) = \frac{\int_{f_1}^{f_2} |X(f)|^2 df}{\int_{f_{min}}^{f_{max}} |X(f)|^2 df}
$$

```python
bp = BandPowerExtractor(sfreq=160, 
                        bands={'mu': (8, 13), 'beta': (13, 30)},
                        relative=True)
X_bp = bp.transform(X)  # 118 features (59 canales × 2 bandas)
```

**Features totales:** CSP (6) + Band Power (118) = **124 features** por época

### 5.4 Modelos de Clasificación

Se implementaron tres algoritmos de Machine Learning tradicional:

#### 1. Linear Discriminant Analysis (LDA)

Clasificador lineal bayesiano que asume distribuciones gaussianas [14]:

$$
y = \arg\max_k \left( \mathbf{w}_k^T \mathbf{x} + b_k \right)
$$

**Ventajas:** Rápido, robusto con pocas muestras, baseline estándar en BCI  
**Desventajas:** Limitado a fronteras lineales

#### 2. Support Vector Machine (SVM)

Clasificador que maximiza el margen entre clases [15]:

- **SVM-Linear:** Kernel lineal, rápido para tiempo real
- **SVM-RBF:** Kernel gaussiano, captura relaciones no lineales

$$
K_{RBF}(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2\right)
$$

**Hiperparámetros:** C=1.0, gamma='scale'

#### 3. Random Forest

Ensemble de 100 árboles de decisión con max_depth=10:

$$
\hat{y} = \text{mode}\{h_1(\mathbf{x}), h_2(\mathbf{x}), ..., h_{100}(\mathbf{x})\}
$$

**Ventajas:** Proporciona importancia de features, robusto a overfitting

### 5.5 Estrategia de Validación

Se utilizó **Leave-One-Run-Out (LORO) Cross-Validation** intra-sujeto [16]:

- **6 folds:** Uno por cada run (R04, R06, R08, R10, R12, R14)
- **Entrenamiento:** 5 runs (5/6 de datos)
- **Test:** 1 run (1/6 de datos)
- **Repetición:** Rotar hasta evaluar todos los runs

**Justificación:** LORO simula un escenario realista donde el modelo se entrena con sesiones previas del usuario y predice en sesiones nuevas, evaluando la estabilidad temporal del clasificador.

**Métricas evaluadas:**
- Accuracy: Proporción de predicciones correctas
- Precision: Predicciones positivas correctas
- Recall: Cobertura de casos positivos reales
- F1-Score: Media armónica de precision y recall
- Cohen's Kappa: Acuerdo ajustado por azar

---

## Resultados

### 6.1 Resultados: Preprocesamiento {#resultados-preprocesamiento}

El pipeline de preprocesamiento procesó exitosamente **109 sujetos** del dataset PhysioNet, generando:

- **Total de épocas:** ~9,700 epochs de imaginación motora
- **Canales procesados:** 59-60 (algunos sujetos con canales removidos por ruido)
- **Duración de épocas:** 577-721 muestras (recortadas a mínimo común: 577)
- **Distribución de clases:**
  - Manos: Izquierda vs Derecha (clasificación binaria)
  - Puños/Pies: Ambas manos vs Ambos pies (clasificación binaria)

El filtrado pasa-banda (8-30 Hz) y la remoción de artefactos mediante ICA mejoraron significativamente el SNR de las señales, como se observa en las Figuras 1 y 2.

### 6.2 Modelos Binarios

Se entrenaron **4 modelos** para cada una de las **2 tareas binarias** (total: 8 modelos):

#### Tarea 1: Clasificación de Manos (Izquierda vs Derecha)

| Modelo | Accuracy | Precision | Recall | F1-Score | Kappa |
|--------|----------|-----------|--------|----------|-------|
| LDA | 60.96% | 60.95% | 60.96% | 60.95% | 0.219 |
| **SVM-RBF** | **62.70%** | **62.68%** | **62.70%** | **62.68%** | **0.254** |
| SVM-Linear | 61.66% | 61.63% | 61.66% | 61.63% | 0.233 |
| Random Forest | 60.23% | 60.18% | 60.23% | 60.18% | 0.205 |

#### Tarea 2: Clasificación de Puños/Pies (Ambas Manos vs Ambos Pies)

| Modelo | Accuracy | Precision | Recall | F1-Score | Kappa |
|--------|----------|-----------|--------|----------|-------|
| LDA | 61.14% | 61.12% | 61.14% | 61.12% | 0.223 |
| **SVM-RBF** | **61.67%** | **61.63%** | **61.67%** | **61.63%** | **0.233** |
| SVM-Linear | 61.63% | 61.55% | 61.63% | 61.55% | 0.233 |
| Random Forest | 60.10% | 60.09% | 60.10% | 60.09% | 0.202 |

![Matrices de Confusión - Manos](/Proyecto/Software/reports/figures/confusion_matrices_hands.png)
*Figura 4: Matrices de confusión normalizadas para clasificación de manos (Izq vs Der). SVM-RBF muestra mejor balance entre clases.*

![Matrices de Confusión - Puños/Pies](/Proyecto/Software/reports/figures/confusion_matrices_fists_feet.png)
*Figura 5: Matrices de confusión normalizadas para clasificación puños/pies (Ambas Manos vs Ambos Pies).*

**Observaciones clave:**

1. **SVM-RBF** alcanzó el mejor desempeño en ambas tareas (62.7% y 61.67%)
2. Los tres modelos principales (LDA, SVM-RBF, SVM-Linear) muestran performance similar (~61-63%)
3. Random Forest fue ligeramente inferior pero aún competitivo (~60%)
4. **Cohen's Kappa >0.2** indica acuerdo sustancial más allá del azar en todos los modelos

![Comparación de Modelos](/Proyecto/Software/reports/figures/model_comparison.png)
*Figura 6: Comparación de métricas entre modelos. SVM-RBF consistentemente superior en ambas tareas.*

![Variabilidad por Fold](/Proyecto/Software/reports/figures/fold_variability.png)
*Figura 7: Variabilidad del accuracy a través de los 6 folds LORO. La estabilidad indica robustez temporal.*

### 6.3 Modelo Multi-Clase

Se entrenó un **modelo unificado de 4 clases** (Mano Izq, Mano Der, Ambas Manos, Ambos Pies) usando Multi-Class CSP (One-vs-Rest):

| Modelo | Accuracy | Precision | Recall | F1-Score | Kappa |
|--------|----------|-----------|--------|----------|-------|
| **LDA** | **38.89%** | **38.86%** | **38.84%** | **38.51%** | **0.185** |
| SVM-RBF | 16.89% | 26.49% | 16.91% | 18.62% | -0.110 |
| SVM-Linear | 36.89% | 36.75% | 36.83% | 36.64% | 0.159 |
| Random Forest | 25.78% | 25.05% | 25.73% | 25.11% | 0.010 |

![Matrices de Confusión 4 Clases](/Proyecto/Software/reports/figures/fig1_confusion_matrices_4class.png)
*Figura 8: Matrices de confusión para modelo de 4 clases. LDA muestra mejor distribución, aunque accuracy es menor que modelos binarios.*

![Comparación Modelos 4 Clases](/Proyecto/Software/reports/figures/fig2_model_comparison_4class.png)
*Figura 9: Comparación de métricas en modelo multi-clase. Mayor dificultad evidente vs clasificación binaria.*

![Variabilidad 4 Clases](/Proyecto/Software/reports/figures/fig3_fold_variability_4class.png)
*Figura 10: Variabilidad por fold en modelo de 4 clases. Mayor inestabilidad refleja complejidad del problema.*

![Tabla Resultados 4 Clases](/Proyecto/Software/reports/figures/fig4_results_table_4class.png)
*Figura 11: Tabla resumen de resultados para modelo multi-clase.*

**Observaciones clave:**

1. **Accuracy ~39%** con LDA, significativamente inferior a modelos binarios (~62%)
2. **SVM-RBF falló** en clasificación multi-clase (16.89%), posible overfitting
3. La complejidad del problema de 4 clases limita el desempeño de ML tradicional
4. **Kappa bajo** (~0.18) indica dificultad para superar predicción aleatoria (25%)

### 6.4 Comparación de Enfoques

| Criterio | 2 Modelos Binarios | 1 Modelo 4 Clases |
|----------|-------------------|-------------------|
| **Mejor Accuracy** | 62.70% (SVM-RBF) | 38.89% (LDA) |
| **Ventaja** | +23.81 puntos | - |
| **Complejidad Arquitectura** | 2 modelos separados | 1 modelo unificado |
| **Features Totales** | 124 (6 CSP + 118 BP) | 142 (24 CSP + 118 BP) |
| **Tiempo de Inferencia** | 2× (cascada) | 1× |
| **Generalidad** | Requiere contexto | Universal |
| **Interpretabilidad** | Alta (binario claro) | Media (4 clases confusas) |
| **Aplicabilidad Clínica** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Análisis del Trade-off:**

- **Modelos Binarios:** Óptimos para aplicaciones donde el contexto es conocido (e.g., usuario indica si quiere mover manos o pies)
- **Modelo Multi-Clase:** Más general pero requiere más datos y técnicas avanzadas (Deep Learning) para alcanzar performance aceptable

---

## Conclusiones

### Logros Principales

1. **Pipeline Automático Completo:** Se desarrolló exitosamente un sistema modular que procesa datos crudos EEG hasta clasificación final, escalable a 109 sujetos y ~9,700 epochs

2. **Resultados Competitivos con ML Tradicional:** El modelo SVM-RBF alcanzó **62.7% de accuracy** en clasificación binaria, comparable a reportes de literatura usando métodos clásicos (60-65%) [17]

3. **Validación Rigurosa:** LORO cross-validation de 6 folds demostró estabilidad temporal de los clasificadores, con variabilidad aceptable entre runs

4. **Comparación de Enfoques:** Se caracterizó empíricamente el trade-off entre precisión (modelos binarios) y generalidad (modelo multi-clase), con diferencia de **23.81 puntos porcentuales** a favor de binarios

5. **Preprocesamiento Efectivo:** La combinación de filtrado pasa-banda, ICA y normalización mejoró significativamente el SNR de las señales EEG

### Limitaciones

1. **Accuracy Moderado (62.7%):** Insuficiente para aplicaciones críticas que requieren >90% de confiabilidad (e.g., control de silla de ruedas autónoma)

2. **Validación Intra-Sujeto:** LORO no evalúa generalización inter-sujeto; se requiere Leave-One-Subject-Out (LOSO) para validación robusta

3. **Clasificación Multi-Clase Limitada:** 38.89% accuracy no es útil para aplicaciones prácticas; requiere Deep Learning o más datos

4. **Sin Optimización de Hiperparámetros:** Se usaron parámetros por defecto; GridSearchCV podría mejorar 5-10%

5. **No Implementado en Tiempo Real:** El sistema actual procesa datos offline; falta pipeline de streaming y feedback

### Contribuciones

- **Código Open-Source:** Pipeline completo reproducible disponible en GitHub con documentación exhaustiva
- **Metodología Comparativa:** Análisis cuantitativo de modelos binarios vs multi-clase en BCI motor imagery
- **Base para Trabajo Futuro:** Arquitectura modular facilita integración de Deep Learning y optimizaciones

### Trabajo Futuro

#### Corto Plazo
1. **Tuning de Hiperparámetros:** GridSearchCV para optimizar C, gamma en SVM
2. **Feature Selection:** Mutual Information o RFE para reducir dimensionalidad
3. **Validación LOSO:** Evaluar generalización inter-sujeto

#### Medio Plazo
4. **Deep Learning:** Implementar EEGNet [7] o Deep ConvNet [8] para mejorar multi-clase
5. **Transfer Learning:** Pre-entrenar en todos los sujetos, fine-tune individual
6. **Data Augmentation:** Time jittering, amplitude scaling para aumentar diversidad

#### Largo Plazo
7. **Sistema en Tiempo Real:** Pipeline de streaming con latencia <300ms
8. **Feedback Adaptativo:** Actualización incremental del modelo por sesión
9. **Validación Clínica:** Pruebas con usuarios con discapacidad motora real

### Reflexión Final

Este proyecto demuestra la viabilidad de sistemas BCI basados en Machine Learning tradicional para clasificación binaria de imaginación motora, alcanzando **62.7% de accuracy** con SVM-RBF. Si bien este desempeño es prometedor para investigación, se requiere incorporar técnicas de Deep Learning y aumentar el dataset para alcanzar el umbral de **70-80%** necesario para aplicaciones clínicas reales. El trade-off entre modelos especializados (binarios) y generales (multi-clase) sugiere una arquitectura híbrida como solución práctica: un clasificador jerárquico que primero identifica el tipo de movimiento (manos/pies) y luego aplica modelos binarios especializados.

---

## Referencias

[1] Wolpaw, J.R., Birbaumer, N., McFarland, D.J., Pfurtscheller, G., Vaughan, T.M. (2002). "Brain-computer interfaces for communication and control." *Clinical Neurophysiology*, 113(6), 767-791.

[2] Ramadan, R.A., Vasilakos, A.V. (2017). "Brain computer interface: control signals review." *Neurocomputing*, 223, 26-44.

[3] Pfurtscheller, G., Neuper, C. (2001). "Motor imagery and direct brain-computer communication." *Proceedings of the IEEE*, 89(7), 1123-1134.

[4] McFarland, D.J., Miner, L.A., Vaughan, T.M., Wolpaw, J.R. (2000). "Mu and beta rhythm topographies during motor imagery and actual movements." *Brain Topography*, 12(3), 177-186.

[5] Ramoser, H., Muller-Gerking, J., Pfurtscheller, G. (2000). "Optimal spatial filtering of single trial EEG during imagined hand movement." *IEEE Transactions on Rehabilitation Engineering*, 8(4), 441-446.

[6] Lotte, F., Congedo, M., Lécuyer, A., Lamarche, F., Arnaldi, B. (2007). "A review of classification algorithms for EEG-based brain-computer interfaces." *Journal of Neural Engineering*, 4(2), R1-R13.

[7] Lawhern, V.J., Solon, A.J., Waytowich, N.R., Gordon, S.M., Hung, C.P., Lance, B.J. (2018). "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces." *Journal of Neural Engineering*, 15(5), 056013.

[8] Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., Ball, T. (2017). "Deep learning with convolutional neural networks for EEG decoding and visualization." *Human Brain Mapping*, 38(11), 5391-5420.

[9] Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M., Rakotomamonjy, A., Yger, F. (2018). "A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update." *Journal of Neural Engineering*, 15(3), 031005.

[10] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004). "BCI2000: a general-purpose brain-computer interface (BCI) system." *IEEE Transactions on Biomedical Engineering*, 51(6), 1034-1043.

[11] Neuper, C., Wörtz, M., Pfurtscheller, G. (2006). "ERD/ERS patterns reflecting sensorimotor activation and deactivation." *Progress in Brain Research*, 159, 211-222.

[12] Pfurtscheller, G., Lopes da Silva, F.H. (1999). "Event-related EEG/MEG synchronization and desynchronization: basic principles." *Clinical Neurophysiology*, 110(11), 1842-1857.

[13] Delorme, A., Makeig, S. (2004). "EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis." *Journal of Neuroscience Methods*, 134(1), 9-21.

[14] Blankertz, B., Lemm, S., Treder, M., Haufe, S., Müller, K.R. (2011). "Single-trial analysis and classification of ERP components—a tutorial." *NeuroImage*, 56(2), 814-825.

[15] Cortes, C., Vapnik, V. (1995). "Support-vector networks." *Machine Learning*, 20(3), 273-297.

[16] Combrisson, E., Jerbi, K. (2015). "Exceeding chance level by chance: The caveat of theoretical chance levels in brain signal classification and statistical assessment of decoding accuracy." *Journal of Neuroscience Methods*, 250, 126-136.

[17] Ang, K.K., Chin, Z.Y., Wang, C., Guan, C., Zhang, H. (2012). "Filter bank common spatial pattern algorithm on BCI competition IV datasets 2a and 2b." *Frontiers in Neuroscience*, 6, 39.

------

*Documento generado como parte del Proyecto Final del curso Introducción a Señales Biomédicas*  
*Universidad Peruana Cayetano Heredia - 2025*
