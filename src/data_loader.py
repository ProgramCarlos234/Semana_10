from __future__ import annotations

# Se importa dataclass para devolver una estructura clara con los datos cargados.
from dataclasses import dataclass

# Se importa NumPy para anotar los arrays del dataset.
import numpy as np
# Se importa el dataset Fashion MNIST desde Keras.
from tensorflow.keras.datasets import fashion_mnist


# Esta clase agrupa de forma explícita las particiones del dataset.
@dataclass(slots=True)
class FashionMnistData:
    """Representar en una sola estructura los datos de entrenamiento y prueba."""

    # Se almacena la matriz de imágenes de entrenamiento.
    train_images: np.ndarray
    # Se almacenan las etiquetas enteras de entrenamiento.
    train_labels: np.ndarray
    # Se almacena la matriz de imágenes de prueba.
    test_images: np.ndarray
    # Se almacenan las etiquetas enteras de prueba.
    test_labels: np.ndarray


# Esta función descarga o recupera el dataset completo Fashion MNIST.
def load_fashion_mnist() -> FashionMnistData:
    """Cargar Fashion MNIST desde TensorFlow/Keras y devolverlo estructurado."""

    # Se obtiene la partición de entrenamiento y la partición de prueba desde Keras.
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # Se empaqueta todo el contenido en una estructura explícita y autoexplicativa.
    return FashionMnistData(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )