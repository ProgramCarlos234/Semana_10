from __future__ import annotations

# Se importa Sequential para crear un modelo secuencial de Keras.
from tensorflow.keras import Sequential
# Se importan las capas necesarias para la arquitectura exigida.
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D

# Se importan constantes de configuración del modelo.
from config.settings import INPUT_SHAPE, LOSS_NAME, METRICS, NUM_CLASSES, OPTIMIZER_NAME


# Esta función construye y compila exactamente la CNN definida en el plan.
def build_cnn_model() -> Sequential:
    """Crear la CNN del proyecto con la arquitectura y compilación aprobadas."""

    # Se define un modelo secuencial con la arquitectura pedida en el plan.
    model = Sequential(
        [
            Input(shape=INPUT_SHAPE),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    # Se compila el modelo con Adam, entropía cruzada dispersa y accuracy.
    model.compile(
        optimizer=OPTIMIZER_NAME,
        loss=LOSS_NAME,
        metrics=METRICS,
    )
    # Se devuelve el modelo listo para entrenar.
    return model