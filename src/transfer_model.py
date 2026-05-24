from __future__ import annotations

# Se importan regularizadores para controlar el sobreajuste en la cabeza final.
from tensorflow.keras import Model
# Se importan las capas necesarias para la arquitectura de transferencia.
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
# Se importa MobileNetV2 como extractor base ligero y eficaz.
from tensorflow.keras.applications import MobileNetV2
# Se importa el regularizador L2 recomendado.
from tensorflow.keras.regularizers import l2

# Se importan constantes de configuración del proyecto.
from config.settings import TRANSFER_DROPOUT_RATE, TRANSFER_INPUT_SHAPE, TRANSFER_L2_FACTOR


# Esta función construye el modelo de transferencia con una cabeza personalizable.
def build_transfer_model(num_classes: int) -> tuple[Model, Model]:
    """Crear MobileNetV2 con cabeza densa para clasificar un dataset propio."""

    # Se define explícitamente la entrada RGB de 224x224 píxeles.
    inputs = Input(shape=TRANSFER_INPUT_SHAPE, name="imagen_entrada")
    # Se carga MobileNetV2 con pesos de ImageNet sin la cabeza final.
    base_model = MobileNetV2(
        input_shape=TRANSFER_INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    # En la primera fase se congela completamente el extractor de características.
    base_model.trainable = False
    # Se conectan las entradas del modelo a la base convolucional.
    features = base_model(inputs, training=False)
    # Se agregan las activaciones espaciales mediante promedio global.
    pooled_features = GlobalAveragePooling2D(name="global_average_pooling")(features)
    # Se añade dropout para reducir el sobreajuste en datasets pequeños.
    regularized_features = Dropout(TRANSFER_DROPOUT_RATE, name="dropout_cabeza")(pooled_features)
    # Se define la capa de clasificación final adaptada al número de clases reales.
    outputs = Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=l2(TRANSFER_L2_FACTOR),
        name="predicciones",
    )(regularized_features)
    # Se construye el modelo funcional completo.
    model = Model(inputs=inputs, outputs=outputs, name="fashion_transfer_mobilenetv2")
    # Se devuelven tanto el modelo final como la base para facilitar el fine-tuning.
    return model, base_model