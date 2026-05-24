from __future__ import annotations

# Se importa Path para admitir rutas de imagen y modelo.
from pathlib import Path

# Se importa NumPy para calcular rankings de probabilidades.
import numpy as np
# Se importa load_model para recuperar el modelo entrenado.
from tensorflow.keras.models import load_model

# Se importan rutas y nombres de clases por defecto del proyecto.
from config.settings import CLASS_NAMES, CLASS_NAMES_PATH, CUSTOM_CLASS_NAMES_PATH, CUSTOM_MODEL_PATH, MODEL_PATH
# Se importa el preprocesamiento de imágenes externas del modelo base.
from src.preprocessing import preprocess_user_image
# Se importa el preprocesamiento robusto para imágenes reales del modo transferencia.
from src.preprocessing_real import preprocess_real_image
# Se importa la carga flexible de JSON.
from src.utils import load_json


# Esta función determina qué nombres de clase corresponden al modelo cargado.
def load_class_names_for_model(model_path: Path) -> list[str]:
    """Recuperar los nombres de clase adecuados al modelo simple o de transferencia."""

    # Si se trata del modelo de dataset propio y existe su JSON asociado, se usa ese fichero.
    if model_path == CUSTOM_MODEL_PATH and CUSTOM_CLASS_NAMES_PATH.exists():
        return list(load_json(CUSTOM_CLASS_NAMES_PATH))
    # Si existe el JSON general del proyecto, se utiliza para el modo Fashion MNIST.
    if CLASS_NAMES_PATH.exists():
        return list(load_json(CLASS_NAMES_PATH))
    # Como fallback final se devuelven los nombres fijos del problema original.
    return CLASS_NAMES


# Esta función detecta si el modelo pertenece al flujo de transferencia.
def is_transfer_model(model_path: Path) -> bool:
    """Indicar si la ruta recibida corresponde al modelo entrenado con dataset propio."""

    # Se compara contra la ruta configurada del modelo de transferencia.
    return model_path == CUSTOM_MODEL_PATH


# Esta función ejecuta una predicción completa sobre una imagen concreta.
def predict_image(image_path: str | Path, model_path: Path = MODEL_PATH) -> dict:
    """Predecir una imagen externa y devolver clase principal y top-3."""

    # Se comprueba que el modelo exista antes de intentar cargarlo.
    if not model_path.exists():
        # Se lanza un error claro para que la GUI o la CLI puedan informar al usuario.
        raise FileNotFoundError(f"No se ha encontrado el modelo en: {model_path}")
    # Se carga el modelo entrenado desde disco.
    model = load_model(model_path)
    # Se cargan los nombres de clase asociados al modelo activo.
    class_names = load_class_names_for_model(model_path)
    # Se decide el preprocesamiento en función del tipo de modelo.
    processed_image = (
        preprocess_real_image(image_path)
        if is_transfer_model(model_path)
        else preprocess_user_image(image_path)
    )
    # Se calculan las probabilidades de pertenencia a cada clase.
    probabilities = model.predict(processed_image, verbose=0)[0]
    # Se obtiene el índice de la clase con mayor probabilidad.
    predicted_index = int(np.argmax(probabilities))
    # Se ordenan todas las clases de mayor a menor probabilidad.
    ranked_predictions = sorted(
        [
            {
                "indice": class_index,
                "clase": class_names[class_index],
                "probabilidad": float(probability),
            }
            for class_index, probability in enumerate(probabilities)
        ],
        key=lambda item: item["probabilidad"],
        reverse=True,
    )
    # Se construye el diccionario final de salida.
    result = {
        "clase_predicha": class_names[predicted_index],
        "indice_predicho": predicted_index,
        "probabilidad": float(probabilities[predicted_index]),
        "top_3": ranked_predictions[:3],
        "modelo_utilizado": "transfer_learning" if is_transfer_model(model_path) else "cnn_simple",
    }
    # Se devuelve la salida preparada para CLI, GUI o informes.
    return result