from __future__ import annotations

# Se importa Path para admitir una ruta alternativa al modelo.
from pathlib import Path

# Se importa NumPy para convertir probabilidades en etiquetas predichas.
import numpy as np
# Se importa pandas para exportar la matriz de confusión a CSV.
import pandas as pd
# Se importan las métricas necesarias desde scikit-learn.
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# Se importa TensorFlow para normalizar el dataset personalizado antes de evaluar.
import tensorflow as tf
# Se importa load_model para recuperar el modelo ya entrenado.
from tensorflow.keras.models import load_model

# Se importan rutas y nombres de clases desde la configuración central.
from config.settings import (
    CLASS_NAMES,
    CONFUSION_MATRIX_CSV_PATH,
    CONFUSION_MATRIX_FIGURE_PATH,
    CUSTOM_CLASSIFIED_EXAMPLES_FIGURE_PATH,
    CUSTOM_CLASS_NAMES_PATH,
    CUSTOM_CONFUSION_MATRIX_CSV_PATH,
    CUSTOM_CONFUSION_MATRIX_FIGURE_PATH,
    CUSTOM_EVALUATION_METRICS_PATH,
    CUSTOM_MODEL_PATH,
    CUSTOM_ROC_AUC_FIGURE_PATH,
    EVALUATION_METRICS_PATH,
    MODEL_PATH,
    CLASSIFIED_EXAMPLES_FIGURE_PATH,
)
# Se importa la carga del dataset.
from src.custom_data_loader import load_custom_image_datasets
from src.data_loader import load_fashion_mnist
# Se importa la normalización de las imágenes de prueba.
from src.preprocessing import normalize_images
# Se importan utilidades de persistencia.
from src.utils import ensure_directory, load_json, save_json
# Se importan funciones de visualización para los resultados.
from src.visualizer import plot_classified_examples, plot_confusion_matrix, plot_multiclass_roc_auc


# Esta función evalúa el modelo guardado y genera todos los reportes asociados.
def evaluate_saved_model(model_path: Path = MODEL_PATH) -> dict:
    """Evaluar el modelo entrenado, guardar métricas y exportar figuras de evaluación."""

    # Se comprueba que el modelo exista antes de continuar.
    if not model_path.exists():
        # Se lanza un error explícito para evitar evaluaciones vacías o engañosas.
        raise FileNotFoundError(f"No se ha encontrado el modelo en: {model_path}")
    # Se cargan los datos completos del problema.
    dataset = load_fashion_mnist()
    # Se normaliza el conjunto de prueba con el mismo criterio del entrenamiento.
    x_test = normalize_images(dataset.test_images)
    # Se cargan los pesos y la arquitectura guardados previamente.
    model = load_model(model_path)
    # Se calcula la pérdida y la accuracy sobre el conjunto de prueba.
    loss, accuracy = model.evaluate(x_test, dataset.test_labels, verbose=1)
    # Se obtienen las probabilidades por clase para cada ejemplo de prueba.
    probabilities = model.predict(x_test, verbose=0)
    # Se escoge la clase de mayor probabilidad para cada ejemplo.
    predicted_labels = np.argmax(probabilities, axis=1)
    # Se calcula la matriz de confusión completa.
    matrix = confusion_matrix(dataset.test_labels, predicted_labels)
    # Se calcula el informe detallado por clase en formato diccionario.
    report = classification_report(
        dataset.test_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    # Se garantiza que la carpeta de reportes exista.
    ensure_directory(CONFUSION_MATRIX_CSV_PATH.parent)
    # Se transforma la matriz en DataFrame etiquetado con nombres de clases.
    matrix_frame = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
    # Se guarda la matriz de confusión como CSV.
    matrix_frame.to_csv(CONFUSION_MATRIX_CSV_PATH, encoding="utf-8")
    # Se guarda también una visualización PNG de la matriz.
    plot_confusion_matrix(matrix, CONFUSION_MATRIX_FIGURE_PATH)
    # Se genera una figura con ejemplos clasificados correctamente.
    plot_classified_examples(
        images=dataset.test_images,
        labels=dataset.test_labels,
        predictions=predicted_labels,
        output_path=CLASSIFIED_EXAMPLES_FIGURE_PATH,
    )
    # Se empaquetan todas las métricas principales para persistencia e informes.
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "total_muestras_test": int(len(dataset.test_labels)),
        "classification_report": report,
    }
    # Se guardan las métricas agregadas en formato JSON.
    save_json(EVALUATION_METRICS_PATH, metrics)
    # Se devuelven las métricas para uso inmediato por CLI u otros módulos.
    return metrics


# Esta función evalúa el modelo de transferencia entrenado con dataset propio.
def evaluate_custom_model(
    model_path: Path = CUSTOM_MODEL_PATH,
    data_dir: str | Path | None = None,
) -> dict:
    """Evaluar el modelo de transferencia y generar métricas avanzadas por clase."""

    # Se comprueba que el modelo exista antes de continuar.
    if not model_path.exists():
        raise FileNotFoundError(f"No se ha encontrado el modelo en: {model_path}")
    # Se cargan los datasets del directorio personalizado.
    datasets = load_custom_image_datasets(data_dir) if data_dir is not None else load_custom_image_datasets()
    # Se obtienen los nombres de clase persistidos o detectados en el dataset.
    class_names = (
        list(load_json(CUSTOM_CLASS_NAMES_PATH))
        if CUSTOM_CLASS_NAMES_PATH.exists()
        else datasets.class_names
    )
    # Se normaliza el conjunto de validación con preprocess_input antes de evaluar.
    prepared_validation_dataset = datasets.validation_dataset.map(
        lambda images, labels: (
            tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(images, tf.float32)),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    # Se carga el modelo entrenado desde disco.
    model = load_model(model_path)
    # Se evalúa el modelo sobre el conjunto de validación.
    loss, accuracy = model.evaluate(prepared_validation_dataset, verbose=1)
    # Se calculan las probabilidades por clase.
    probabilities = model.predict(prepared_validation_dataset, verbose=0)
    # Se recuperan las etiquetas reales lote a lote.
    true_labels = np.concatenate([labels.numpy() for _, labels in datasets.validation_dataset], axis=0)
    # Se obtienen las predicciones discretas.
    predicted_labels = np.argmax(probabilities, axis=1)
    # Se calcula la matriz de confusión.
    matrix = confusion_matrix(true_labels, predicted_labels)
    # Se calcula el informe detallado por clase.
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    # Se transforma y guarda la matriz de confusión tabular.
    matrix_frame = pd.DataFrame(matrix, index=class_names, columns=class_names)
    ensure_directory(CUSTOM_CONFUSION_MATRIX_CSV_PATH.parent)
    matrix_frame.to_csv(CUSTOM_CONFUSION_MATRIX_CSV_PATH, encoding="utf-8")
    # Se genera la visualización de la matriz de confusión.
    plot_confusion_matrix(matrix, CUSTOM_CONFUSION_MATRIX_FIGURE_PATH, class_names=class_names, title="Matriz de confusión del modelo de transferencia")
    # Se recuperan imágenes y etiquetas para ilustrar ejemplos clasificados.
    sample_images = np.concatenate([images.numpy() for images, _ in datasets.validation_dataset], axis=0)
    plot_classified_examples(
        images=sample_images,
        labels=true_labels,
        predictions=predicted_labels,
        output_path=CUSTOM_CLASSIFIED_EXAMPLES_FIGURE_PATH,
        class_names=class_names,
        title="Ejemplos clasificados por el modelo de transferencia",
        cmap=None,
    )
    # Se calcula el ROC-AUC multiclase cuando existan al menos dos clases y muestras suficientes.
    one_hot_labels = np.eye(len(class_names))[true_labels]
    roc_auc_macro = float(roc_auc_score(one_hot_labels, probabilities, multi_class="ovr", average="macro"))
    roc_curves: list[dict[str, object]] = []
    for class_index, class_name in enumerate(class_names):
        false_positive_rate, true_positive_rate, _ = roc_curve(one_hot_labels[:, class_index], probabilities[:, class_index])
        roc_curves.append(
            {
                "class_name": class_name,
                "fpr": false_positive_rate.tolist(),
                "tpr": true_positive_rate.tolist(),
            }
        )
    plot_multiclass_roc_auc(roc_curves, CUSTOM_ROC_AUC_FIGURE_PATH)
    # Se empaquetan las métricas principales del modelo profesional.
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "roc_auc_macro_ovr": roc_auc_macro,
        "total_muestras_validacion": int(len(true_labels)),
        "class_names": class_names,
        "classification_report": report,
    }
    # Se guardan las métricas en JSON para análisis posterior.
    save_json(CUSTOM_EVALUATION_METRICS_PATH, metrics)
    # Se devuelven las métricas calculadas.
    return metrics