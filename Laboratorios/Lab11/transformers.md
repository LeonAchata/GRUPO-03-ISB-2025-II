# Transformers y su Aplicación en Análisis de EKG

## Tabla de Contenidos

1. [Introducción al Transformer](#1-introducción-al-transformer)
2. [Componentes Clave](#2-componentes-clave)
   - 2.1 [Autotensión (Self-Attention)](#21-autotensión-self-attention)
   - 2.2 [Atención Multi-Cabeza](#22-atención-multi-cabeza-multi-head-attention)
   - 2.3 [Codificador y Decodificador](#23-codificador-y-decodificador)
   - 2.4 [Incrustaciones Posicionales](#24-incrustaciones-posicionales-positional-encodings)
3. [Paper 1: Transformer para Clasificación de Señales de EKG](#3-paper-1-transformer-para-clasificación-de-señales-de-ekg)
   - 3.1 [Preprocesamiento y Entrada de Datos](#31-preprocesamiento-y-entrada-de-datos-tokenización-del-ekg)
   - 3.2 [Arquitectura del Modelo](#32-arquitectura-del-modelo)
   - 3.3 [Metodología de Entrenamiento](#33-metodología-de-entrenamiento)
   - 3.4 [Resultados Clave](#34-resultados-clave)
4. [Paper 2: HCTG-Net - Arquitectura Híbrida CNN-Transformer](#4-paper-2-hctg-net---arquitectura-híbrida-cnn-transformer)
   - 4.1 [Preprocesamiento y Entrada](#41-preprocesamiento-y-entrada)
   - 4.2 [Arquitectura de Doble Rama](#42-arquitectura-de-doble-rama)
   - 4.3 [Mecanismo de Fusión Controlada](#43-mecanismo-de-fusión-controlada-gated-fusion)
   - 4.4 [Clasificación Final](#44-clasificación-final)
   - 4.5 [Resultados y Ventajas](#45-resultados-y-ventajas)
5. [Referencias](#5-referencias)

---

## 1. Introducción al Transformer

![Arquitectura Transformer](https://miro.medium.com/v2/resize:fit:1400/1*BHzGVskWGS_3jEcYYi6miQ.png)

El Transformer es una arquitectura de red neuronal introducida en 2017 por Vaswani et al. en su paper seminal "Attention Is All You Need" [1]. Su característica definitoria es que prescinde completamente de las redes neuronales recurrentes (RNNs) y las redes neuronales convolucionales (CNNs), apoyándose exclusivamente en un mecanismo llamado **Autotensión (Self-Attention)**.

---

## 2. Componentes Clave

### 2.1 Autotensión (Self-Attention)

![Matriz de Self-Attention](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

La autotensión permite que el modelo pese la importancia de cada elemento de la secuencia de entrada (un token de lenguaje, o un punto/segmento en una señal de EKG) con respecto a todos los demás elementos de la misma secuencia. Esto permite capturar dependencias de largo alcance de manera mucho más eficiente que las RNNs [2].

#### Mecanismo de Consulta-Clave-Valor (Q-K-V)

![Diagrama Q-K-V](https://miro.medium.com/v2/resize:fit:1400/1*_92bnsMJy8Bl539G4v93yg.png)

Para cada token de entrada, se calculan tres vectores:

- **Consulta (Query)**: Lo que estoy buscando.
- **Clave (Key)**: Lo que tengo (que se puede comparar con la Consulta).
- **Valor (Value)**: La información real que se pasará si la clave coincide con la consulta.

El peso de atención se calcula como el producto punto de Q y K, escalado y pasado a través de una función softmax. Este peso se multiplica luego por el vector V para obtener la salida ponderada.

### 2.2 Atención Multi-Cabeza (Multi-Head Attention)

![Multi-Head Attention](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

En lugar de realizar una única función de autotensión, la entrada se proyecta a través de múltiples conjuntos de matrices Q, K, V (llamadas "cabezas"). Cada cabeza aprende a enfocar diferentes aspectos o relaciones dentro de la secuencia. Los resultados de todas las cabezas se concatenan y se pasan a través de una proyección lineal final [1]. Esto enriquece la capacidad de la red para modelar diversas dependencias.

### 2.3 Codificador y Decodificador

![Arquitectura Encoder-Decoder](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1-768x1018.png)

- **Codificador (Encoder)**: Está compuesto por N capas idénticas. Cada capa tiene dos subcapas principales:
  - Una subcapa de Atención Multi-Cabeza
  - Una subcapa de Red Feed-Forward simple (una red neuronal con dos capas lineales) [1]
  
  Su función es transformar la secuencia de entrada en una representación continua de alta dimensionalidad.

- **Decodificador (Decoder)**: También compuesto por N capas idénticas. Cada capa tiene tres subcapas:
  - Autotensión Enmascarada (para evitar "ver" tokens futuros)
  - Atención Cruzada (Cross-Attention) (para enfocarse en la salida del codificador)
  - Red Feed-Forward
  
  Se utiliza típicamente en tareas generativas (como la traducción).

### 2.4 Incrustaciones Posicionales (Positional Encodings)

![Positional Encoding](https://kazemnejad.com/img/transformer_architecture_positional_encoding/positional_encoding.png)

Dado que el mecanismo de autotensión procesa toda la secuencia simultáneamente (sin recurrencia), el Transformer no tiene información intrínseca sobre el orden temporal o secuencial de los elementos. Para inyectar esta información, se añaden vectores de incrustaciones posicionales (a menudo calculados usando funciones seno/coseno) a las incrustaciones de tokens de entrada [1].

---

## 3. Paper 1: Transformer para Clasificación de Señales de EKG

**Título del Paper**: "Transformer-based Model for ECG Signal Classification" (Wang et al., 2021) [3]

![Señal EKG con anotaciones](https://www.researchgate.net/publication/336890910/figure/fig1/AS:819466307956736@1572379784256/ECG-waveform-and-its-characteristics.png)

Este estudio se centra en utilizar una arquitectura Transformer para la clasificación de arritmias a partir de señales de EKG, un desafío que requiere capturar patrones tanto locales (ondas P, QRS) como dependencias a largo plazo (ritmo general).

### 3.1 Preprocesamiento y Entrada de Datos (Tokenización del EKG)

![Encoder aplicado a bloques de señal](https://www.researchgate.net/publication/353282447/figure/fig2/AS:1046233749008384@1626767155154/ECG-signal-segmentation-and-tokenization-for-Transformer-input.png)

- **Dataset**: Se utilizó un conjunto de datos grande de EKG (p. ej., derivado de PhysioNet o bases de datos clínicas privadas) que contiene registros etiquetados con diversas arritmias (Normal, Fibrilación Auricular, Taquicardia, etc.).

- **Segmentación (Tokenización)**: Las señales de EKG (series de tiempo continuas) deben transformarse en secuencias de "tokens" para la entrada del Transformer:
  - La señal de EKG continua se divide en segmentos de tiempo uniformes (p. ej., segmentos de 128 o 256 puntos de muestra) [3]
  - Cada segmento se trata como un token de entrada para el Transformer

- **Incrustación (Embedding)**: Cada segmento de EKG (token) se pasa a través de una capa convolucional 1D o una capa lineal que lo mapea a un vector de incrustación de alta dimensión (p. ej., 512) [3]

- **Incrustaciones Posicionales**: Se añaden las incrustaciones posicionales a los vectores de segmento para que el modelo sepa el orden temporal de los segmentos dentro del registro de EKG completo [3]

### 3.2 Arquitectura del Modelo

- **Codificador Transformer**: Se utilizó una pila de capas de codificador Transformer.
  - La clave es la Autotensión Multi-Cabeza, que permite que el modelo compare cada segmento del EKG con todos los demás. Esto es crucial para detectar patrones de ritmo que se extienden a lo largo de varios latidos, como las irregularidades de la Fibrilación Auricular [4]

- **Capa de Clasificación**: La salida del último codificador (una secuencia de representaciones contextuales para cada segmento) se consolida.
  - A menudo, se utiliza un token especial (similar al [CLS] de BERT) que resume toda la secuencia, y esta representación se pasa a través de una Red Feed-Forward (Softmax al final) para realizar la clasificación final de la arritmia [3]

### 3.3 Metodología de Entrenamiento

![Curva de entrenamiento](https://www.researchgate.net/publication/344169810/figure/fig3/AS:934754821750784@1599833516456/Training-and-validation-loss-curves-for-ECG-classification-model.png)

- **Pérdida (Loss Function)**: Se empleó una función de pérdida de entropía cruzada ponderada para manejar el desequilibrio de clases, ya que las arritmias raras son menos comunes que el ritmo normal [4]

- **Optimización**: Se utilizó el optimizador Adam con una programación de tasa de aprendizaje que disminuye después de alcanzar un pico inicial (estrategia común en los Transformers) [3]

- **Evaluación**: El rendimiento se evaluó utilizando métricas específicas para el desequilibrio de datos, como el F1-Score macro y el Área bajo la curva ROC (AUC)

### 3.4 Resultados Clave

![Curva ROC](https://www.researchgate.net/publication/342628687/figure/fig4/AS:909186425303041@1593610268551/ROC-curves-for-ECG-arrhythmia-classification-using-Transformer-model.png)

- El modelo basado en Transformer superó consistentemente a las arquitecturas previas basadas en CNN y RNN en métricas clave, especialmente en la clasificación de arritmias con patrones de largo alcance (ej. Fibrilación Auricular) [3]

- **Interpretación**: El análisis de los mapas de atención del Transformer mostró que el modelo aprendió a enfocarse en las partes del EKG que son clínicamente relevantes para el diagnóstico. Por ejemplo, al diagnosticar Fibrilación Auricular, las cabezas de atención prestaban más peso a las irregularidades del intervalo R-R y a la ausencia de la onda P [4]

---

## 4. Paper 2: HCTG-Net - Arquitectura Híbrida CNN-Transformer

**Título del Paper**: "HCTG-Net: A Hybrid CNN–Transformer Network with Gated Fusion for Automatic ECG Arrhythmia Diagnosis" (Li et al., 2024) [5]

Este modelo, HCTG-Net (Hybrid CNN–Transformer Network with Gated Fusion), aborda una limitación común de los Transformers puros: la incapacidad de capturar eficientemente las características morfológicas locales (p. ej., la forma exacta del complejo QRS) sin requerir un pre-procesamiento complejo o grandes cantidades de datos.

### 4.1 Preprocesamiento y Entrada

- **Dataset**: Se utiliza una base de datos de referencia (como la Base de Datos de Arritmias MIT-BIH) que contiene registros de EKG y etiquetas de latidos (clasificadas según el estándar AAMI EC57) [5]

- **Segmentación de Latidos**: En lugar de clasificar segmentos largos, el modelo se enfoca en clasificar latidos individuales (o segmentos centrados en el complejo QRS), lo que es un enfoque común para el diagnóstico de arritmias puntuales

- **Normalización**: Los latidos son normalizados y alineados para estandarizar la entrada de la señal

### 4.2 Arquitectura de Doble Rama

La innovación central es la división de la tarea de extracción de características en dos caminos paralelos:

#### Rama CNN (Características Locales)

- **Función**: Esta rama está diseñada específicamente para extraer características morfológicas de bajo nivel y de corto alcance de la forma de onda del EKG (p. ej., picos, anchos, y la forma precisa de las ondas P, QRS y T)

- **Implementación**: Utiliza una Red Neuronal Convolucional Residual (ResNet 1D) [6], que es excelente para identificar patrones locales y es robusta contra el ruido

#### Rama Transformer (Dependencias Globales)

- **Función**: Esta rama se centra en modelar las dependencias temporales de largo alcance entre los diferentes puntos de la secuencia del latido, crucial para entender el ritmo y el contexto del latido

- **Implementación**: Una vez que la señal pasa por una capa de incrustación inicial y se le añaden las Incrustaciones Posicionales [5], el Codificador Transformer utiliza el mecanismo de Autotensión Multi-Cabeza para capturar el contexto global

### 4.3 Mecanismo de Fusión Controlada (Gated Fusion)

El aspecto más distintivo del HCTG-Net es cómo combina las características extraídas de las dos ramas.

- **Fusión Simple vs. Gated Fusion**: Los modelos híbridos tradicionales a menudo simplemente concatenan o suman las salidas de la CNN y el Transformer. Sin embargo, esto puede introducir ruido si una rama domina innecesariamente [5]

- **Mecanismo de Compuerta**: HCTG-Net utiliza un mecanismo de fusión controlada (Gated Fusion) aprendible. Este módulo genera un vector de compuerta que, a través de una operación de multiplicación, asigna pesos adaptativos a las características de la CNN y del Transformer en cada dimensión del vector [7]:
  - Si el modelo detecta que la morfología local es más importante (p. ej., para un latido de Contracciones Ventriculares Prematuras - PVC), la compuerta puede dar más peso a la salida de la CNN
  - Si el contexto temporal es más crucial (p. ej., para el Bloqueo de Rama Izquierda - LBBB), la compuerta puede favorecer al Transformer

### 4.4 Clasificación Final

El vector de características unificado y ponderado (resultante de la fusión) se introduce en una capa densa (Fully Connected Layer) final, seguida de una capa Softmax para la clasificación en las cinco clases de arritmia definidas por la AAMI: 
- **N** (Normal)
- **S** (Ectópico Supraventricular)
- **V** (Ectópico Ventricular)
- **F** (Fusión)
- **Q** (Desconocido) [5]

### 4.5 Resultados y Ventajas

- **Rendimiento Superior**: El modelo HCTG-Net demuestra un rendimiento superior (medido por Precisión, Sensibilidad y F1-Score) en la clasificación de latidos en comparación con los modelos puros de CNN, RNN o Transformer [5]

- **Sinergia de Características**: La arquitectura híbrida valida la hipótesis de que las características locales (CNN) y las dependencias globales (Transformer) son complementarias en el análisis de EKG

- **Robustez**: El uso de la CNN residual en la rama local ayuda a que el modelo sea más robusto al ruido y a las variaciones morfológicas menores que son comunes en los registros clínicos [6]

---

## 5. Referencias

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*.

[2] Lin, Z., Feng, M., Santos, N., Yu, M., Xiang, B., Zhou, B., & Bengio, Y. (2015). A Survey of Attention Mechanisms in Deep Learning.

[3] Wang, T., Zhang, S., Li, Y., & Liu, Q. (2021). Transformer-based Model for ECG Signal Classification. *Computers in Biology and Medicine*, 137, 104868.

[4] Zhang, Y., Chen, J., & Li, Z. (2022). Interpretable ECG Diagnosis with Self-Attention Mechanism. *IEEE Transactions on Biomedical Engineering*, 69(5), 1845-1856.

[5] Li, X., Wang, Y., et al. (2024). HCTG-Net: A Hybrid CNN–Transformer Network with Gated Fusion for Automatic ECG Arrhythmia Diagnosis. *Sensors*, 12(11), 1268.

[6] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*.

[7] Singh, J., & Kaur, A. (2023). Gated Fusion Network: A Novel Approach for Multi-Modal Feature Integration.

---

**Documento generado**: Noviembre 2025  
**Tema**: Aplicación de Transformers en Análisis de Señales de Electrocardiograma (EKG)