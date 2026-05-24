# Plan detallado del proyecto: Sistema inteligente de clasificación automática de prendas con CNN y Fashion MNIST

## 1. Objetivo del proyecto

Diseñar y desarrollar un sistema académico completo en **Python** para la **clasificación automática de prendas de vestir** usando **Deep Learning con una red neuronal convolucional (CNN)** y el dataset **Fashion MNIST**, aplicando de forma explícita la metodología **CRISP-DM** en sus seis fases.

El sistema deberá incluir:

- entrenamiento y evaluación de una CNN funcional con **TensorFlow/Keras**;
- cálculo y visualización de métricas (**accuracy, loss, matriz de confusión**);
- gráficos del proceso de entrenamiento;
- predicción sobre una **imagen propia del usuario**;
- una **interfaz gráfica en Python** para uso demostrativo;
- documentación académica completa, redactada en **español de España**.

## 2. Estructura propuesta del proyecto

```text
fashion/
├── PLAN_PROYECTO.md
├── README.md
├── requirements.txt
├── main.py
├── config/
│   └── settings.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_user_images/
├── models/
│   ├── trained/
│   │   ├── fashion_cnn.keras
│   │   └── class_names.json
│   └── reports/
│       ├── training_history.csv
│       ├── confusion_matrix.csv
│       └── evaluation_metrics.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_builder.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── predictor.py
│   ├── visualizer.py
│   ├── utils.py
│   └── gui/
│       ├── __init__.py
│       ├── app.py
│       ├── image_handlers.py
│       └── widgets.py
├── docs/
│   ├── memoria_academica.md
│   ├── presentacion_resultados.md
│   ├── figuras/
│   └── tablas/
└── notebooks/
    └── exploratory_analysis.ipynb
```

## 3. Arquitectura propuesta de la CNN

### Capas propuestas

1. **Conv2D**: 32 filtros, kernel `(3,3)`, activación `relu`
2. **MaxPooling2D**: tamaño `(2,2)`
3. **Conv2D**: 64 filtros, kernel `(3,3)`, activación `relu`
4. **MaxPooling2D**: tamaño `(2,2)`
5. **Flatten**
6. **Dense**: 128 neuronas, activación `relu`
7. **Dropout**: 0.3
8. **Dense**: 10 neuronas, activación `softmax`

### Diagrama textual

```text
Entrada (28x28x1)
   ↓
Conv2D(32, 3x3, ReLU)
   ↓
MaxPooling2D(2x2)
   ↓
Conv2D(64, 3x3, ReLU)
   ↓
MaxPooling2D(2x2)
   ↓
Flatten
   ↓
Dense(128, ReLU)
   ↓
Dropout(0.3)
   ↓
Dense(10, Softmax)
   ↓
Clase predicha
```

## 4. Flujo de datos

1. Cargar Fashion MNIST desde `tensorflow.keras.datasets`.
2. Normalizar los valores de píxel.
3. Reestructurar imágenes a formato `(n, 28, 28, 1)`.
4. Construir y compilar la CNN.
5. Entrenar el modelo y registrar métricas.
6. Evaluar el conjunto de prueba.
7. Generar matriz de confusión y gráficos.
8. Guardar el modelo y artefactos.
9. Permitir al usuario cargar una imagen propia.
10. Preprocesar la imagen externa y ejecutar inferencia.

## 5. Fases CRISP-DM

| Fase | Qué se hará | Archivos principales |
|---|---|---|
| Comprensión del negocio | Definir problema, objetivos y alcance | `PLAN_PROYECTO.md`, `docs/memoria_academica.md` |
| Comprensión de los datos | Analizar Fashion MNIST y sus clases | `notebooks/exploratory_analysis.ipynb` |
| Preparación de los datos | Normalización y adaptación de entradas | `src/data_loader.py`, `src/preprocessing.py` |
| Modelado | Construcción y entrenamiento de la CNN | `src/model_builder.py`, `src/trainer.py` |
| Evaluación | Métricas, gráficas y matriz de confusión | `src/evaluator.py`, `src/visualizer.py` |
| Despliegue | Integración del modelo en GUI | `main.py`, `src/gui/app.py`, `src/predictor.py` |

## 6. Módulos Python y responsabilidades

- `config/settings.py`: constantes, hiperparámetros y rutas.
- `src/data_loader.py`: carga del dataset.
- `src/preprocessing.py`: normalización y tratamiento de imágenes.
- `src/model_builder.py`: definición del modelo CNN.
- `src/trainer.py`: entrenamiento y guardado del modelo.
- `src/evaluator.py`: evaluación, matriz de confusión y métricas.
- `src/predictor.py`: predicción sobre imágenes individuales.
- `src/visualizer.py`: gráficos de entrenamiento.
- `src/gui/app.py`: ventana principal.
- `src/gui/image_handlers.py`: carga y adaptación de imágenes del usuario.
- `src/gui/widgets.py`: componentes reutilizables de la interfaz.

## 7. Diseño de la GUI

- Ventana principal con título del proyecto.
- Botón para cargar imagen.
- Área de vista previa.
- Botón de predicción.
- Resultado principal con clase predicha.
- Lista de probabilidades principales.
- Mensajes de error o validación.

## 8. Decisiones técnicas clave

| Decisión | Valor propuesto | Justificación |
|---|---|---|
| Tamaño de entrada | `28x28x1` | Coincide con Fashion MNIST |
| Batch size | `32` | Equilibrio entre estabilidad y consumo |
| Épocas | `10` | Base razonable para una primera versión |
| Optimizador | `Adam` | Buen rendimiento con poca configuración |
| Función de pérdida | `sparse_categorical_crossentropy` | Adecuada para etiquetas enteras |
| GUI | `Tkinter` | Ligera e integrada en Python |

## 9. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Sobreajuste | Uso de dropout y control de épocas |
| Bajo rendimiento con fotos reales | Preprocesamiento y explicación de limitaciones |
| Interfaz poco robusta | Validaciones y separación por módulos |
| Proyecto demasiado amplio | Priorizar versión base funcional |

## 10. Criterios de aceptación

1. El modelo entrena correctamente con Fashion MNIST.
2. Se muestran `accuracy`, `loss` y matriz de confusión.
3. Se generan gráficos de entrenamiento.
4. El modelo puede predecir una imagen propia del usuario.
5. La GUI permite cargar una imagen y mostrar la predicción.
6. La documentación académica cubre todos los apartados exigidos.
7. El proyecto se mantiene en español y con estructura modular.