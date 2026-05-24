from __future__ import annotations

# Se importa Path para aceptar rutas de imagen externas.
from pathlib import Path

# Se importa NumPy para operar con tensores e imágenes.
import numpy as np
# Se importan utilidades de Pillow para conversión y tratamiento de imágenes.
from PIL import Image, ImageOps

# Se importan constantes de dimensiones del proyecto.
from config.settings import IMAGE_HEIGHT, IMAGE_WIDTH


# Esta función normaliza imágenes y añade el canal requerido por la CNN.
def normalize_images(images: np.ndarray) -> np.ndarray:
    """Escalar los píxeles al rango [0, 1] y adaptar la forma a CNN."""

    # Se convierte el array a coma flotante para asegurar una normalización correcta.
    images_float = images.astype("float32")
    # Se dividen todos los valores de píxel entre 255 para llevarlos al rango [0, 1].
    images_scaled = images_float / 255.0
    # Se añade una dimensión final de canal para obtener tensores de forma (n, 28, 28, 1).
    images_with_channel = np.expand_dims(images_scaled, axis=-1)
    # Se devuelve el conjunto de imágenes ya adaptado para la red convolucional.
    return images_with_channel


# Esta función prepara simultáneamente entrenamiento y prueba.
def prepare_datasets(train_images: np.ndarray, test_images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalizar y adaptar a CNN las particiones de entrenamiento y prueba."""

    # Se normaliza el conjunto de entrenamiento.
    x_train = normalize_images(train_images)
    # Se normaliza el conjunto de prueba.
    x_test = normalize_images(test_images)
    # Se devuelven ambas particiones ya preparadas.
    return x_train, x_test


# Esta función preprocesa una imagen externa del usuario para inferencia.
def preprocess_user_image(image_path: str | Path) -> np.ndarray:
    """Convertir una imagen externa a escala de grises, 28x28 y normalizarla."""

    # Se abre la imagen original desde la ruta indicada.
    image = Image.open(image_path)
    # Se convierte la imagen a escala de grises porque Fashion MNIST trabaja con un solo canal.
    image_gray = ImageOps.grayscale(image)
    # Se calcula la intensidad media para decidir si conviene invertir la imagen.
    mean_intensity = float(np.array(image_gray, dtype="float32").mean())
    # Se invierte la imagen sólo cuando el fondo es claramente claro y la prenda oscura.
    adjusted_image = ImageOps.invert(image_gray) if mean_intensity > 127.0 else image_gray
    # Se redimensiona la imagen al tamaño exacto esperado por la red.
    image_resized = adjusted_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    # Se transforma la imagen redimensionada a un array NumPy en coma flotante.
    image_array = np.array(image_resized, dtype="float32")
    # Se normalizan los píxeles al rango [0, 1].
    image_scaled = image_array / 255.0
    # Se añade la dimensión de canal.
    image_with_channel = np.expand_dims(image_scaled, axis=-1)
    # Se añade la dimensión de lote porque Keras espera lotes incluso para una sola imagen.
    image_batch = np.expand_dims(image_with_channel, axis=0)
    # Se devuelve el tensor listo para inferencia.
    return image_batch


# Esta función transforma un array 28x28 del dataset en una imagen PNG guardable.
def dataset_image_to_pil(image_array: np.ndarray, upscale_factor: int = 10) -> Image.Image:
    """Convertir una imagen del dataset en una imagen PIL ampliada para demostración."""

    # Se crea un objeto PIL a partir del array en escala de grises.
    image = Image.fromarray(image_array.astype("uint8"), mode="L")
    # Se amplía la imagen para que sea más fácil de visualizar fuera del dataset.
    enlarged_image = image.resize(
        (IMAGE_WIDTH * upscale_factor, IMAGE_HEIGHT * upscale_factor),
        resample=Image.Resampling.NEAREST,
    )
    # Se devuelve la imagen generada.
    return enlarged_image