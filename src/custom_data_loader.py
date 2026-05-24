from __future__ import annotations

# Se importa Counter para calcular pesos de clase de forma trazable.
from collections import Counter
# Se importa dataclass para devolver una estructura clara.
from dataclasses import dataclass
# Se importa Path para manipular rutas del dataset.
from pathlib import Path

# Se importa NumPy para cálculos sobre distribuciones de etiquetas.
import numpy as np
# Se importa TensorFlow para construir datasets eficientes.
import tensorflow as tf

# Se importan parámetros globales de configuración.
from config.settings import (
    BATCH_SIZE,
    CUSTOM_DATASET_DIR,
    CUSTOM_VALIDATION_SPLIT,
    RANDOM_SEED,
    TRANSFER_IMAGE_HEIGHT,
    TRANSFER_IMAGE_WIDTH,
)


# Esta estructura agrupa todos los componentes del dataset propio.
@dataclass(slots=True)
class CustomImageDatasets:
    """Representar los datasets y metadatos del modo con imágenes reales."""

    # Se almacena el dataset de entrenamiento.
    train_dataset: tf.data.Dataset
    # Se almacena el dataset de validación usado también para evaluación inicial.
    validation_dataset: tf.data.Dataset
    # Se guardan los nombres de las carpetas-clase detectadas.
    class_names: list[str]
    # Se guarda el número total de clases.
    num_classes: int
    # Se almacenan los pesos de clase recomendados.
    class_weight: dict[int, float]
    # Se registra el directorio raíz utilizado.
    data_dir: Path


# Esta función valida la estructura tipo ImageFolder antes de cargarla.
def validate_custom_dataset_dir(data_dir: str | Path = CUSTOM_DATASET_DIR) -> Path:
    """Comprobar que el dataset existe y contiene carpetas de clase válidas."""

    # Se normaliza la ruta recibida a objeto Path.
    dataset_path = Path(data_dir)
    # Si no existe la carpeta, se interrumpe con un mensaje accionable.
    if not dataset_path.exists():
        raise FileNotFoundError(
            "No se ha encontrado el directorio del dataset propio. "
            f"Crea la ruta {dataset_path} con subcarpetas por categoría."
        )
    # Se localizan únicamente subdirectorios directos que representen clases.
    class_directories = sorted([path for path in dataset_path.iterdir() if path.is_dir()])
    # Si no hay al menos dos clases, no es posible entrenar un clasificador útil.
    if len(class_directories) < 2:
        raise ValueError(
            "El dataset propio debe contener al menos dos carpetas de clase. "
            "Ejemplo: data/custom_dataset/camisa/*.jpg"
        )
    # Se devuelve la ruta validada para reutilización posterior.
    return dataset_path


# Esta función calcula pesos inversamente proporcionales a la frecuencia por clase.
def compute_class_weight_from_dataset(dataset: tf.data.Dataset, num_classes: int) -> dict[int, float]:
    """Calcular class_weight a partir de las etiquetas presentes en el dataset."""

    # Se inicializa un contador de frecuencias por índice de clase.
    counter: Counter[int] = Counter()
    # Se recorren los lotes de entrenamiento para contar etiquetas.
    for _, labels in dataset:
        # Se actualiza el contador con las etiquetas enteras del lote actual.
        counter.update(int(label) for label in labels.numpy().tolist())
    # Se calcula el número total de muestras observadas.
    total_samples = sum(counter.values())
    # Se construyen pesos equilibrados por frecuencia inversa.
    class_weight = {
        class_index: float(total_samples / (num_classes * max(counter.get(class_index, 1), 1)))
        for class_index in range(num_classes)
    }
    # Se devuelve el diccionario listo para model.fit.
    return class_weight


# Esta función carga el dataset propio desde directorios con un split fijo de validación.
def load_custom_image_datasets(data_dir: str | Path = CUSTOM_DATASET_DIR) -> CustomImageDatasets:
    """Crear datasets de entrenamiento y validación desde una estructura ImageFolder."""

    # Se valida primero la estructura básica del dataset.
    dataset_path = validate_custom_dataset_dir(data_dir)
    # Se construye el conjunto de entrenamiento con un split reproducible.
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="int",
        validation_split=CUSTOM_VALIDATION_SPLIT,
        subset="training",
        seed=RANDOM_SEED,
        image_size=(TRANSFER_IMAGE_HEIGHT, TRANSFER_IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    # Se construye el conjunto de validación correspondiente.
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="int",
        validation_split=CUSTOM_VALIDATION_SPLIT,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=(TRANSFER_IMAGE_HEIGHT, TRANSFER_IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    # Se extraen los nombres de clase detectados por Keras.
    class_names = list(train_dataset.class_names)
    # Se calcula el número total de clases del problema.
    num_classes = len(class_names)
    # Se optimiza la canalización para entrenamiento y evaluación.
    autotune = tf.data.AUTOTUNE
    optimized_train_dataset = train_dataset.cache().prefetch(buffer_size=autotune)
    optimized_validation_dataset = validation_dataset.cache().prefetch(buffer_size=autotune)
    # Se calculan pesos de clase para datasets pequeños o desbalanceados.
    class_weight = compute_class_weight_from_dataset(train_dataset, num_classes)
    # Se devuelven datasets y metadatos agrupados.
    return CustomImageDatasets(
        train_dataset=optimized_train_dataset,
        validation_dataset=optimized_validation_dataset,
        class_names=class_names,
        num_classes=num_classes,
        class_weight=class_weight,
        data_dir=dataset_path,
    )


# Esta función resume rápidamente la distribución por clases del dataset original.
def summarize_dataset_structure(data_dir: str | Path = CUSTOM_DATASET_DIR) -> dict[str, int]:
    """Obtener el número de imágenes por carpeta para informar al usuario."""

    # Se valida la ruta y la estructura del dataset.
    dataset_path = validate_custom_dataset_dir(data_dir)
    # Se inicializa el resumen por nombre de clase.
    summary: dict[str, int] = {}
    # Se recorren las carpetas de clase una a una.
    for class_dir in sorted([path for path in dataset_path.iterdir() if path.is_dir()]):
        # Se cuentan extensiones de imagen habituales de forma no recursiva.
        image_files = [
            file_path
            for file_path in class_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        # Se guarda el recuento por categoría.
        summary[class_dir.name] = int(len(image_files))
    # Se devuelve el resumen para CLI, README o memoria.
    return summary