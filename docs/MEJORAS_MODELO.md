# Mejoras profesionales del modelo para clasificación de imágenes reales de ropa

## Introducción

El estado actual del proyecto ofrece un rendimiento correcto sobre **Fashion MNIST** con una `accuracy` cercana al 90 %, pero presenta una limitación estructural habitual: el modelo ha aprendido sobre imágenes **pequeñas, centradas, en escala de grises y con fondo controlado**, mientras que las imágenes reales que subirá el usuario serán **RGB, con fondos variados, iluminación no uniforme, ángulos distintos y mayor variabilidad intra-clase**. Por ello, la mejora profesional no debe centrarse en “forzar” la CNN simple, sino en **preparar una tubería moderna de visión por computador** que sea capaz de generalizar mejor cuando el usuario incorpore su propio dataset.

Este documento resume las recomendaciones técnicas para llevar el sistema desde una solución académica base hasta un enfoque mucho más sólido para clasificación de ropa en imágenes reales. El objetivo no es entrenar todavía con un dataset propio, sino dejar el proyecto **estructurado, documentado y listo** para ello.

---

## 1. Dataset: calidad, balanceo, mínimo de imágenes y estructura recomendada

### Explicación

La calidad del dataset es el factor que más condiciona la capacidad de generalización del modelo. En clasificación de ropa, no basta con tener imágenes “bonitas”: hace falta que las muestras representen la variabilidad real del problema. Eso incluye:

- distintos fondos;
- diferentes condiciones de luz;
- variedad de tallas, tejidos, pliegues y poses;
- múltiples ángulos;
- prendas solas y sin objetos distractores dominantes.

Con sólo unas **10 imágenes por categoría**, el modelo puede entrenar técnicamente, pero el riesgo de sobreajuste será muy alto. En un escenario profesional ligero, la recomendación razonable es:

- **mínimo aceptable**: `30-50` imágenes por clase si se usa transferencia y augmentation fuerte;
- **recomendado**: `100-500` imágenes por clase;
- **óptimo para mayor robustez**: más de `500` imágenes por clase cuando se desee distinguir categorías visualmente próximas.

También es clave el **balanceo entre clases**. Si una clase tiene 300 imágenes y otra sólo 20, el modelo tenderá a favorecer la clase mayoritaria. En esos casos conviene:

- recopilar más imágenes en las clases deficitarias;
- usar `class_weight`;
- reforzar augmentation en clases conflictivas.

La estructura de carpetas debe seguir el patrón tipo **ImageFolder**, ya que simplifica la carga automática:

```text
data/custom_dataset/
├── camisa/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── pantalon/
│   ├── img_001.jpg
│   └── ...
├── vestido/
└── zapatillas/
```

Finalmente, un error muy frecuente consiste en que el modelo aprenda el **fondo** en lugar de la prenda. Si todas las camisas se fotografían sobre una mesa blanca y todos los pantalones sobre una cama oscura, la red puede correlacionar fondo con etiqueta. Eso degrada la generalización fuera del entorno de captura.

### Recomendación

- Usar un mínimo de `100-500` imágenes por clase si el objetivo es clasificar fotos reales con robustez razonable.
- Si al principio sólo se dispone de `10-30` imágenes por clase, usar **transfer learning + augmentation + class_weight**, asumiendo que será una fase preliminar.
- Mantener fondo lo más neutro posible o aplicar segmentación previa.
- Variar condiciones de captura para que el modelo no dependa del contexto.

### Código de ejemplo

```python
from pathlib import Path

dataset_root = Path("data/custom_dataset")

for class_dir in sorted(dataset_root.iterdir()):
    if class_dir.is_dir():
        image_count = len(
            [
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
        )
        print(f"{class_dir.name}: {image_count} imágenes")
```

### Justificación técnica

En problemas de visión, los modelos no aprenden “conceptos semánticos” como lo haría un humano; aprenden correlaciones estadísticas. Si el dataset no representa bien la distribución real de entrada, el modelo optimizará sobre patrones espurios y fallará cuando cambie el dominio. Por eso el diseño del dataset es la primera decisión profesional.

---

## 2. Preprocesamiento: resize 224x224, normalización, centrado, iluminación y ruido

### Explicación

Para usar **MobileNetV2** o arquitecturas equivalentes, es necesario adaptar la entrada a `224x224` en RGB. Además, la normalización no debe hacerse con una simple división entre 255 cuando se trabaja con un modelo preentrenado, sino mediante `preprocess_input`, que ajusta los valores al rango esperado por la red.

En imágenes reales de ropa también es útil:

- **centrar o recortar** la prenda para reducir fondo irrelevante;
- mejorar la iluminación con **CLAHE**;
- reducir ruido con filtros suaves;
- corregir orientación EXIF;
- mantener consistencia geométrica antes del redimensionado.

### Recomendación

- Redimensionar siempre a `224x224` si se usa MobileNetV2.
- Aplicar `preprocess_input`.
- Centrar la prenda sobre un lienzo cuadrado antes del resize.
- Usar mejora de contraste local con CLAHE y denoising ligero cuando la calidad de captura sea irregular.

### Código de ejemplo

```python
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

image = Image.open("mi_prenda.jpg").convert("RGB")
image = ImageOps.exif_transpose(image)
image = ImageOps.contain(image, (224, 224))

canvas = Image.new("RGB", (224, 224), color=(255, 255, 255))
offset = ((224 - image.width) // 2, (224 - image.height) // 2)
canvas.paste(image, offset)

image_array = np.array(canvas, dtype=np.float32)
image_batch = np.expand_dims(image_array, axis=0)
image_batch = preprocess_input(image_batch)
```

### Justificación técnica

La transferencia de aprendizaje sólo funciona bien si las entradas siguen el contrato de preprocesado de la arquitectura base. Además, centrar la prenda reduce la varianza no informativa. CLAHE ayuda cuando hay imágenes oscuras o con contraste desigual, y la reducción de ruido evita que la red responda a artefactos de compresión o captura.

---

## 3. Data augmentation específico para ropa

### Explicación

El *data augmentation* debe ser realista. En ropa, algunas transformaciones tienen sentido y otras no. Los parámetros recomendados son:

- `rotation_range = 15-20` grados;
- `zoom = 0.1-0.2`;
- `horizontal flip = sí`;
- `vertical flip = no`;
- brillo entre `0.8` y `1.2`;
- contraste moderado;
- `width_shift` y `height_shift = 0.1`.

### Recomendación

- Rotación moderada para simular pequeñas variaciones de captura.
- Zoom ligero para tolerar distintos encuadres.
- Flip horizontal sí, porque una prenda suele seguir siendo la misma visualmente.
- Flip vertical no, porque invertir una prenda boca abajo produce muestras poco naturales.
- Ajustes moderados de brillo y contraste para robustez ante iluminación variable.
- Desplazamientos suaves para evitar dependencia del centrado perfecto.

### Código de ejemplo

```python
import tensorflow as tf

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15)),
        tf.keras.layers.RandomContrast(0.2),
    ]
)
```

### Justificación técnica

El augmentation actúa como regularizador implícito. En datasets pequeños, obliga al modelo a aprender patrones más estables y menos dependientes de la geometría exacta de cada ejemplo. Sin embargo, si se exagera, se generan ejemplos artificiales que empeoran el aprendizaje. Por eso los rangos deben ser prudentes y consistentes con la física del problema.

---

## 4. Transfer Learning con MobileNetV2 en dos fases

### Explicación

La mejora más importante para imágenes reales consiste en reemplazar la CNN simple por un modelo preentrenado. **MobileNetV2** es una opción excelente cuando se busca equilibrio entre:

- tamaño del modelo;
- velocidad de inferencia;
- consumo de memoria;
- capacidad para generalizar en imágenes reales.

La estrategia recomendada es en **dos fases**:

1. **Fase 1**: congelar toda la base convolucional y entrenar sólo la cabeza de clasificación.
2. **Fase 2**: descongelar las últimas `20-30` capas y hacer *fine-tuning* con tasa de aprendizaje muy baja (`1e-5`).

### Recomendación

- Usar `GlobalAveragePooling2D + Dropout + Dense(softmax)`.
- Entrenar primero sólo la cabeza.
- Después ajustar finamente las últimas capas profundas.

### Código de ejemplo

```python
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2

inputs = Input(shape=(224, 224, 3))
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
```

**Fine-tuning:**

```python
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

### Justificación técnica

Las capas iniciales de modelos entrenados en ImageNet aprenden patrones universales (bordes, texturas, formas locales). Aprovechar ese conocimiento reduce la necesidad de grandes volúmenes de datos. El fine-tuning profundo sólo debe hacerse al final y con LR bajo para evitar destruir representaciones útiles ya aprendidas.

---

## 5. Overfitting: detección y mitigación

### Explicación

El sobreajuste aparecerá con facilidad si el usuario empieza con un dataset reducido. La señal más clara es el **gap entre entrenamiento y validación**:

- `accuracy_train` sube mucho;
- `accuracy_val` se estanca o cae;
- `loss_val` empeora mientras `loss_train` sigue mejorando.

### Recomendación

- `Dropout` entre `0.3` y `0.5`;
- `EarlyStopping(patience=5)`;
- `ReduceLROnPlateau(factor=0.2, patience=3)`;
- regularización `L2`;
- augmentation consistente;
- observar curvas de entrenamiento en cada ejecución.

### Código de ejemplo

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
    ),
]
```

### Justificación técnica

El objetivo no es optimizar al máximo el conjunto de entrenamiento, sino mejorar la generalización. Early stopping evita seguir aprendiendo ruido. ReduceLROnPlateau ayuda a estabilizar el ajuste fino. Dropout y L2 reducen coadaptación de neuronas, algo muy importante cuando hay pocas muestras por clase.

---

## 6. Entrenamiento: épocas, batch size, LR, optimizador y validación

### Explicación

Los hiperparámetros recomendados para el modo profesional son:

- `batch_size = 32`;
- fase 1: `20-50` épocas;
- fase 2: `10-20` épocas;
- LR fase 1: `1e-3`;
- LR fase 2: `1e-5`;
- optimizador `Adam`;
- `validation_split = 0.2`.

### Recomendación

- Empezar con `30` épocas en fase 1 y `15` en fase 2.
- Mantener `batch_size=32` salvo limitaciones de memoria.
- Reservar `20 %` de validación para medir generalización.

### Código de ejemplo

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

### Justificación técnica

Adam ofrece convergencia estable con poca calibración manual. El split de validación del 20 % es un compromiso razonable en datasets pequeños: deja suficientes datos para entrenar y, al mismo tiempo, permite detectar sobreajuste con cierta fiabilidad.

---

## 7. Categorías visualmente similares

### Explicación

En ropa, algunas clases son especialmente difíciles:

- polo frente a camisa;
- leggings frente a jeans;
- jersey frente a sudadera;
- zapatilla casual frente a deportiva.

### Recomendación

- Aumentar datos de las clases conflictivas.
- Aplicar `class_weight` si hay desbalance.
- Afinar más capas profundas en el fine-tuning.
- Considerar *focal loss* si el error se concentra en categorías concretas.
- Inspeccionar la matriz de confusión en cada entrenamiento.

### Código de ejemplo

```python
class_weight = {
    0: 1.0,
    1: 1.2,
    2: 1.6,
    3: 1.0,
}
```

### Justificación técnica

Cuando las clases comparten siluetas o texturas, la red necesita más diversidad intra-clase y más capacidad discriminativa. El fine-tuning profundo ayuda a especializar rasgos de alto nivel, mientras que los pesos de clase evitan que las categorías escasas queden infrarepresentadas en la optimización.

---

## 8. Imágenes reales: fondo, segmentación y aumentos agresivos

### Explicación

Las imágenes reales suelen introducir:

- fondos complejos;
- oclusiones;
- sombras;
- iluminación heterogénea;
- encuadres parciales.

### Recomendación

- Aplicar preprocesado robusto antes de inferir.
- Usar `rembg` o U2Net si está disponible.
- Como respaldo ligero, emplear `GrabCut` de OpenCV.
- Aplicar augmentation algo más agresivo cuando el dataset real sea pequeño.

### Código de ejemplo

```python
processed = preprocess_real_image(
    image_path="foto_usuario.jpg",
    remove_background=True,
)
```

### Justificación técnica

Reducir la influencia del fondo hace que el clasificador se concentre en contorno, textura y estructura de la prenda. Esto mejora la robustez cuando el usuario captura imágenes en entornos domésticos no controlados.

---

## 9. Arquitectura: MobileNetV2 vs EfficientNetB0 vs ResNet50

### Explicación

Tres arquitecturas razonables para este proyecto son:

| Arquitectura | Ventajas | Inconvenientes | Recomendación |
|---|---|---|---|
| MobileNetV2 | Ligera, rápida, excelente en CPU | Algo menos precisa que modelos mayores | Ideal si no hay GPU |
| EfficientNetB0 | Mejor equilibrio precisión/eficiencia | Más lenta que MobileNetV2 | Recomendación principal con GPU |
| ResNet50 | Robusta y conocida | Más pesada y costosa | Útil si se busca baseline clásico de mayor capacidad |

### Recomendación

- **EfficientNetB0** si se dispone de GPU y se busca un pequeño salto de precisión.
- **MobileNetV2** si el entrenamiento e inferencia van a ejecutarse en CPU o hardware modesto.

### Código de ejemplo

```python
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
```

### Justificación técnica

En un contexto universitario con recursos limitados, el coste computacional es un criterio real. MobileNetV2 ofrece una relación rendimiento/latencia muy buena. EfficientNetB0 suele rendir mejor si hay capacidad de cómputo suficiente, especialmente en tareas con textura y forma complejas como ropa real.

---

## 10. Evaluación: matriz de confusión, métricas por clase, curvas y ROC-AUC

### Explicación

La `accuracy` global es insuficiente para un análisis serio. Deben incorporarse:

- matriz de confusión;
- `precision`, `recall` y `F1` por clase;
- curvas de `accuracy/loss`;
- ROC-AUC multiclase.

### Recomendación

- Generar un `classification_report` tras cada entrenamiento importante.
- Revisar qué clases fallan, no sólo cuántas fallan.
- Guardar figuras para la memoria y para diagnóstico.

### Código de ejemplo

```python
from sklearn.metrics import classification_report, confusion_matrix

report = classification_report(y_true, y_pred, target_names=class_names)
matrix = confusion_matrix(y_true, y_pred)
print(report)
print(matrix)
```

### Justificación técnica

El análisis por clase permite detectar problemas ocultos. Un modelo con 85 % de accuracy puede estar clasificando muy bien clases fáciles y muy mal las conflictivas. La evaluación multiclase completa aporta una visión mucho más profesional del rendimiento real.

---

## 11. GUI: mejoras de usabilidad y comunicación de confianza

### Explicación

Para una aplicación de demostración orientada al usuario final, la interfaz debe:

- mostrar el **top-3** de predicciones;
- incluir barras de confianza en porcentaje;
- permitir limpiar el estado;
- informar cuando la imagen está siendo procesada;
- manejar errores de forma legible.

### Recomendación

- Priorizar el modelo de transferencia si existe.
- Mantener compatibilidad con la CNN simple si aún no se ha entrenado el modelo nuevo.
- Mostrar mensajes claros si falta el dataset o el modelo.
- Considerar drag & drop en una iteración futura, aunque no es imprescindible para la versión actual.

### Código de ejemplo

```python
prediction = predict_image(image_path, model_path=active_model_path)

for item in prediction["top_3"]:
    print(item["clase"], item["probabilidad"] * 100)
```

### Justificación técnica

El top-3 reduce la rigidez de una decisión única y ayuda a interpretar casos ambiguos. Esto es especialmente útil en prendas visualmente parecidas. Además, una interfaz clara mejora la defendibilidad académica y la experiencia de uso.

---

## 12. Objetivo final: checklist de buenas prácticas en visión por computador para ropa

### Checklist recomendada

- [ ] Dataset real con estructura `data/custom_dataset/categoria/*.jpg`
- [ ] Mínimo recomendable de `100-500` imágenes por clase
- [ ] Clases razonablemente balanceadas
- [ ] Fondos variados o segmentación previa
- [ ] Resize a `224x224`
- [ ] Normalización con `preprocess_input`
- [ ] Data augmentation realista para ropa
- [ ] Transfer learning en dos fases
- [ ] EarlyStopping y ReduceLROnPlateau
- [ ] Dropout y L2 para controlar sobreajuste
- [ ] Métricas por clase y matriz de confusión
- [ ] ROC-AUC multiclase
- [ ] GUI con top-3, barras de confianza y errores claros
- [ ] Compatibilidad con el modelo simple como fallback

### Recomendación final

La estrategia profesional más sólida para este proyecto consiste en mantener la CNN simple como referencia académica y, en paralelo, preparar un segundo flujo basado en **transfer learning**, especialmente diseñado para **imágenes reales de ropa**. Esta dualidad permite conservar la trazabilidad pedagógica del proyecto original y, al mismo tiempo, abrir una línea realista de mejora cuando el usuario suba su dataset propio.

---

## Pasos exactos que seguirá el usuario cuando suba sus imágenes

### Estructura exacta de carpetas

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

### Comando exacto de entrenamiento

```bash
python main.py train_custom --data-dir data/custom_dataset
```

### Comando exacto de evaluación

```bash
python main.py evaluate_custom --data-dir data/custom_dataset
```

### Resultado esperado

- modelo guardado en `models/trained/fashion_transfer.keras`;
- nombres de clases en `models/trained/class_names_custom.json`;
- historial en `models/reports/training_history_transfer.csv`;
- métricas en `models/reports/evaluation_metrics_transfer.json`;
- nuevas figuras en `docs/figuras/`.