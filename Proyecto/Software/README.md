# EEG-BCI Motor Imagery Classification Project

## Descripción del Proyecto

Sistema de interfaz cerebro-computadora (BCI) para detectar imaginación motora mediante señales EEG, permitiendo a personas con discapacidades motoras controlar dispositivos a través de la actividad cerebral.

## Objetivos

- Clasificar 4 tipos de imaginación motora: mano izquierda, mano derecha, ambas manos, ambos pies
- Procesamiento de señales EEG en tiempo real
- Lograr >70% de accuracy en clasificación intra-sujeto

## Dataset

- **Fuente**: PhysioNet EEG Motor Movement/Imagery Dataset
- **Participantes**: 109 sujetos
- **Canales**: 64 canales EEG @ 160 Hz
- **Formato**: EDF+ con anotaciones de eventos
- **Runs de interés**: R04, R06, R08, R10, R12, R14 (imaginación motora)

## Estructura del Proyecto

```
Proyecto-ISB/
├── data/                 # Datos EEG (raw + processed)
├── notebooks/            # Jupyter notebooks de análisis
├── src/                  # Código fuente modular
│   ├── data/            # Carga y parsing de datos
│   ├── preprocessing/   # Filtrado y limpieza
│   ├── features/        # Extracción de características
│   ├── models/          # Modelos de clasificación
│   ├── validation/      # Validación cruzada
│   └── utils/           # Utilidades
├── experiments/         # Resultados de experimentos
├── models/             # Modelos entrenados
├── reports/            # Reportes y figuras
└── tests/              # Tests unitarios
```

## Instalación

1. Crear entorno virtual:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Fase 1: Exploración de Datos
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### Fase 2: Preprocesamiento
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### Fase 3: Entrenamiento
```bash
jupyter notebook notebooks/04_model_training.ipynb
```

## Pipeline de Procesamiento

1. **Carga de datos**: Leer archivos EDF y eventos
2. **Filtrado**: Pasa-banda 8-30 Hz (bandas mu/beta)
3. **Segmentación**: Extraer epochs [-0.5s, +4s] desde eventos
4. **Extracción de características**: CSP + PSD
5. **Clasificación**: LDA, SVM, Deep Learning
6. **Validación**: Leave-One-Run-Out (LORO)

## Métricas de Evaluación

- Accuracy, Precision, Recall, F1-Score
- Matriz de confusión
- Latencia de clasificación

## Roadmap

- [x] Fase 1: Configuración y exploración (Semana 1)
- [ ] Fase 2: Preprocesamiento avanzado (Semana 2-3)
- [ ] Fase 3: Modelado y clasificación (Semana 4-5)
- [ ] Fase 4: Evaluación y optimización (Semana 6)

## Referencias

[1] WHO - Spinal Cord Injury (2021)
[2] Instituto Nacional de Rehabilitación - Perú (2024)
[3] PhysioNet EEG Motor Movement/Imagery Dataset

## Autores

Proyecto ISB - PUCP

## Licencia

MIT License
