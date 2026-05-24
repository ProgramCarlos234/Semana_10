from __future__ import annotations

# Se importa TensorFlow para construir una canalización de aumento de datos robusta.
import tensorflow as tf


# Se intenta importar la capa nativa de brillo aleatorio cuando esté disponible.
try:
    # Se importan las capas de aumento recomendadas para imágenes de ropa.
    from tensorflow.keras.layers import (
        RandomBrightness,
        RandomContrast,
        RandomFlip,
        RandomRotation,
        RandomZoom,
    )
# Si la versión instalada no expone RandomBrightness, se define una capa compatible.
except ImportError:
    # Se importan las capas restantes, disponibles de forma estable.
    from tensorflow.keras.layers import RandomContrast, RandomFlip, RandomRotation, RandomZoom

    # Esta clase implementa un ajuste aleatorio de brillo compatible con Keras.
    class RandomBrightness(tf.keras.layers.Layer):
        """Aplicar una variación aleatoria de brillo durante el entrenamiento."""

        # Este método recibe el rango de brillo recomendado para el problema.
        def __init__(self, factor: tuple[float, float] = (-0.2, 0.2), **kwargs) -> None:
            """Inicializar la capa guardando el intervalo de perturbación."""

            # Se inicializa la clase base de Keras.
            super().__init__(**kwargs)
            # Se guardan los límites inferior y superior del cambio de brillo.
            self.factor = factor

        # Este método aplica el aumento sólo mientras el modelo entrena.
        def call(self, inputs, training: bool | None = None):
            """Modificar el brillo de forma aleatoria manteniendo el rango [0, 255]."""

            # Si no se está entrenando, se devuelve la imagen intacta.
            if training is False:
                return inputs
            # Se calcula un delta aleatorio dentro del rango configurado.
            brightness_delta = tf.random.uniform(
                shape=(),
                minval=self.factor[0],
                maxval=self.factor[1],
                dtype=tf.float32,
            )
            # Se transforma temporalmente al rango [0, 1] para usar la primitiva de TensorFlow.
            normalized_inputs = tf.cast(inputs, tf.float32) / 255.0
            # Se aplica el cambio de brillo.
            adjusted_inputs = tf.image.adjust_brightness(normalized_inputs, brightness_delta)
            # Se recorta al rango válido y se vuelve al rango [0, 255].
            clipped_inputs = tf.clip_by_value(adjusted_inputs, 0.0, 1.0) * 255.0
            # Se devuelve el tensor transformado.
            return clipped_inputs

        # Este método permite serializar correctamente la capa personalizada.
        def get_config(self) -> dict:
            """Devolver la configuración necesaria para guardar y restaurar la capa."""

            # Se recupera la configuración base de Keras.
            base_config = super().get_config()
            # Se añade el parámetro propio de esta capa.
            base_config.update({"factor": self.factor})
            # Se devuelve la configuración combinada.
            return base_config


# Esta función construye la secuencia de aumento recomendada para ropa.
def build_clothing_augmentation() -> tf.keras.Sequential:
    """Crear un bloque de data augmentation conservador y realista para prendas."""

    # Se devuelve un Sequential para reutilizarlo fácilmente en el entrenamiento.
    return tf.keras.Sequential(
        [
            RandomFlip(mode="horizontal", name="flip_horizontal"),
            RandomRotation(factor=0.08, fill_mode="nearest", name="rotation_15_20_deg"),
            RandomZoom(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15), fill_mode="nearest", name="zoom_10_20"),
            RandomContrast(factor=0.2, name="contrast_20"),
            RandomBrightness(factor=(-0.2, 0.2), name="brightness_08_12"),
        ],
        name="aumento_prendas",
    )