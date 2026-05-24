# Sistema inteligente de clasificación automática de prendas con CNN y Fashion MNIST

Proyecto académico completo desarrollado en **Python** para clasificar prendas de vestir del dataset **Fashion MNIST** mediante una **red neuronal convolucional (CNN)**. La solución sigue la estructura aprobada del plan, genera artefactos de entrenamiento y evaluación, incorpora una **interfaz gráfica Tkinter** y está documentada en español de España con enfoque universitario.

## Descripción

El sistema resuelve un problema de clasificación multiclase sobre diez categorías de prendas:

1. Camiseta o top
2. Pantalón
3. Jersey
4. Vestido
5. Abrigo
6. Sandalia
7. Camisa
8. Zapatilla deportiva
9. Bolso
10. Botín

La CNN implementada sigue exactamente la arquitectura aprobada:

- `Conv2D(32, 3x3, relu)`
- `MaxPooling2D(2x2)`
- `Conv2D(64, 3x3, relu)`
- `MaxPooling2D(2x2)`
- `Flatten`
- `Dense(128, relu)`
- `Dropout(0.3)`
- `Dense(10, softmax)`

La compilación usa:

- optimizador `Adam`
- función de pérdida `sparse_categorical_crossentropy`
- `batch_size=32`
- `epochs=10`

## Requisitos

- Python 3.10 o superior
- TensorFlow
- NumPy
- Matplotlib
- Pillow
- scikit-learn
- pandas
- OpenCV

`rembg` queda como dependencia **opcional** para eliminación de fondo más robusta en imágenes reales.

## Instalación

```bash
python -m pip install -r requirements.txt
```

## Cómo entrenar

```bash
python main.py train
```

Este comando:

- descarga o reutiliza Fashion MNIST;
- entrena la CNN;
- guarda el modelo en `models/trained/fashion_cnn.keras`;
- guarda el historial en `models/reports/training_history.csv`;
- genera métricas y matriz de confusión;
- crea figuras PNG en `docs/figuras/`;
- genera una imagen de ejemplo en `data/sample_user_images/`;
- registra una predicción demostrativa en `models/reports/sample_prediction.json`.

## Cómo evaluar

```bash
python main.py evaluate
```

Este comando carga el modelo guardado y vuelve a generar:

- `models/reports/evaluation_metrics.json`
- `models/reports/confusion_matrix.csv`
- `docs/figuras/matriz_confusion.png`
- `docs/figuras/ejemplos_clasificados.png`

## Modo dataset propio

El proyecto incorpora un flujo adicional, no destructivo, pensado para clasificar **imágenes reales de ropa** mediante **transfer learning con MobileNetV2**. Este modo convive con la CNN simple original.

### Estructura exacta del dataset

Coloca tus imágenes siguiendo esta estructura:

```text
data/custom_dataset/
├── camisa/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── pantalon/
│   ├── 001.jpg
│   └── ...
├── vestido/
│   └── ...
└── zapatilla/
    └── ...
```

Cada subcarpeta representa una clase. Se recomienda:

- mínimo aceptable: `30-50` imágenes por clase;
- recomendable: `100-500` imágenes por clase;
- fondos variados pero sin que el fondo domine la escena;
- imágenes RGB nítidas y con la prenda razonablemente centrada.

### Entrenar con dataset propio

```bash
python main.py train_custom --data-dir data/custom_dataset
```

Este comando:

- carga el dataset con `image_dataset_from_directory`;
- crea un split de validación del `20 %`;
- aplica *data augmentation* específico para ropa;
- entrena un modelo de transferencia en dos fases;
- guarda el modelo en `models/trained/fashion_transfer.keras`;
- genera métricas, curvas y figuras adicionales.

### Evaluar el modelo de dataset propio

```bash
python main.py evaluate_custom --data-dir data/custom_dataset
```

Este comando vuelve a generar:

- `models/reports/evaluation_metrics_transfer.json`
- `models/reports/confusion_matrix_transfer.csv`
- `docs/figuras/matriz_confusion_transfer.png`
- `docs/figuras/ejemplos_clasificados_transfer.png`
- `docs/figuras/roc_auc_transfer.png`

### Qué modelo usa la GUI

Cuando existe `models/trained/fashion_transfer.keras`, la interfaz gráfica lo prioriza automáticamente para imágenes reales. Si todavía no existe, la GUI mantiene compatibilidad con `models/trained/fashion_cnn.keras`.

## Resultado obtenido en la ejecución validada

En la ejecución real realizada en este entorno se obtuvo:

- `accuracy` de test: `0.9106`
- `loss` de test: `0.2522`
- predicción correcta sobre la imagen de ejemplo generada desde el dataset

## Cómo lanzar la GUI

```bash
python main.py gui
```

La interfaz permite:

- cargar una imagen del usuario;
- ver una previsualización;
- ejecutar la predicción;
- mostrar la clase predicha;
- visualizar el top-3 de probabilidades con barras horizontales;
- limpiar la selección actual;
- mostrar un indicador de estado durante el procesamiento;
- manejar errores si falta el modelo o la imagen no es válida.

## Estructura de carpetas

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
│   ├── custom_dataset/
│   └── sample_user_images/
├── models/
│   ├── trained/
│   └── reports/
├── src/
│   ├── data_loader.py
│   ├── custom_data_loader.py
│   ├── preprocessing.py
│   ├── preprocessing_real.py
│   ├── model_builder.py
│   ├── transfer_model.py
│   ├── augmentation.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── predictor.py
│   ├── visualizer.py
│   ├── utils.py
│   └── gui/
│       ├── app.py
│       ├── image_handlers.py
│       └── widgets.py
├── docs/
│   ├── memoria_academica.md
│   ├── MEJORAS_MODELO.md
│   ├── presentacion_resultados.md
│   ├── figuras/
│   └── tablas/
└── notebooks/
    └── exploratory_analysis.ipynb
```

## Autores

- Proyecto preparado para defensa universitaria en la asignatura de Construcción de Software.
- Desarrollo técnico asistido y documentación final integrada en este repositorio.