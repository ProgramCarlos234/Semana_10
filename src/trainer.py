from __future__ import annotations

# Se importa Path para manipular rutas de archivos de salida.
from pathlib import Path

# Se importa pandas para dejar un registro tabular adicional si fuera necesario.
import pandas as pd
# Se importa TensorFlow para optimizadores, callbacks y capas de preprocesamiento.
import tensorflow as tf

# Se importan constantes de configuración del proyecto.
from config.settings import (
    ACCURACY_FIGURE_PATH,
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    CUSTOM_ACCURACY_FIGURE_PATH,
    CUSTOM_CLASS_NAMES_PATH,
    CUSTOM_DATASET_DIR,
    CUSTOM_EVALUATION_METRICS_PATH,
    CUSTOM_LOSS_FIGURE_PATH,
    CUSTOM_MODEL_PATH,
    CUSTOM_TRAINING_HISTORY_PATH,
    CUSTOM_VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    EVALUATION_METRICS_PATH,
    FIGURES_DIR,
    LOSS_FIGURE_PATH,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REPORTS_DIR,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SAMPLE_IMAGE_PATH,
    SAMPLE_PREDICTION_PATH,
    SAMPLE_USER_IMAGES_DIR,
    TABLES_DIR,
    TRANSFER_FINE_TUNE_LAYERS,
    TRANSFER_PHASE_1_EPOCHS,
    TRANSFER_PHASE_1_LEARNING_RATE,
    TRANSFER_PHASE_2_EPOCHS,
    TRANSFER_PHASE_2_LEARNING_RATE,
    TRAINED_MODELS_DIR,
    TRAINING_HISTORY_PATH,
    VALIDATION_SPLIT,
)
# Se importa el aumento de datos especializado en ropa.
from src.augmentation import build_clothing_augmentation
# Se importa la carga del dataset propio.
from src.custom_data_loader import load_custom_image_datasets, summarize_dataset_structure
# Se importa la carga de datos.
from src.data_loader import load_fashion_mnist
# Se importa la evaluación posterior al entrenamiento.
from src.evaluator import evaluate_custom_model, evaluate_saved_model
# Se importa el constructor del modelo CNN.
from src.model_builder import build_cnn_model
# Se importa la predicción individual para la demostración final.
from src.predictor import predict_image
# Se importa la preparación de datos y la generación de imagen de ejemplo.
from src.preprocessing import dataset_image_to_pil, prepare_datasets
# Se importa el modelo de transferencia basado en MobileNetV2.
from src.transfer_model import build_transfer_model
# Se importan utilidades generales de persistencia y semillas.
from src.utils import ensure_project_directories, save_json, set_global_seed
# Se importan funciones de visualización del entrenamiento.
from src.visualizer import plot_training_curve, save_training_history_csv


# Esta función genera una imagen de muestra a partir del propio dataset de prueba.
def create_sample_user_image(test_images, test_labels) -> dict:
    """Crear y guardar una imagen de ejemplo del dataset para demostrar la predicción."""

    # Se toma un índice estable para disponer siempre del mismo ejemplo demostrativo.
    sample_index = 0
    # Se extrae la imagen del conjunto de prueba.
    sample_image = test_images[sample_index]
    # Se extrae la etiqueta real asociada a la imagen.
    sample_label = int(test_labels[sample_index])
    # Se transforma la imagen 28x28 en una imagen PIL ampliada.
    sample_pil_image = dataset_image_to_pil(sample_image)
    # Se asegura que la carpeta de imágenes de muestra exista.
    ensure_project_directories([SAMPLE_USER_IMAGES_DIR])
    # Se guarda la imagen sintética en la ruta definida por configuración.
    sample_pil_image.save(SAMPLE_IMAGE_PATH)
    # Se devuelve un resumen de la muestra creada.
    return {
        "ruta_imagen": str(SAMPLE_IMAGE_PATH),
        "indice_real": sample_label,
        "clase_real": CLASS_NAMES[sample_label],
    }


# Esta función entrena el modelo, guarda artefactos y ejecuta la evaluación final.
def train_and_save_model() -> dict:
    """Entrenar la CNN, persistir artefactos y ejecutar una evaluación completa."""

    # Se fijan semillas para mejorar la repetibilidad del entrenamiento.
    set_global_seed(RANDOM_SEED)
    # Se aseguran todas las carpetas relevantes del proyecto.
    ensure_project_directories(
        [
            RAW_DATA_DIR,
            PROCESSED_DATA_DIR,
            SAMPLE_USER_IMAGES_DIR,
            TRAINED_MODELS_DIR,
            REPORTS_DIR,
            FIGURES_DIR,
            TABLES_DIR,
        ]
    )
    # Se cargan las particiones originales del dataset Fashion MNIST.
    dataset = load_fashion_mnist()
    # Se preparan los tensores normalizados de entrenamiento y prueba.
    x_train, x_test = prepare_datasets(dataset.train_images, dataset.test_images)
    # Se construye el modelo CNN aprobado en el plan.
    model = build_cnn_model()
    # Se entrena el modelo con validación interna y los hiperparámetros exigidos.
    history = model.fit(
        x_train,
        dataset.train_labels,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    # Se garantiza la existencia de la carpeta de modelos entrenados.
    ensure_project_directories([TRAINED_MODELS_DIR, REPORTS_DIR])
    # Se guarda el modelo entrenado en formato Keras.
    model.save(MODEL_PATH)
    # Se guardan los nombres de las clases para reutilización en GUI e informes.
    save_json(CLASS_NAMES_PATH, CLASS_NAMES)
    # Se guarda el historial de entrenamiento en CSV.
    save_training_history_csv(history.history, TRAINING_HISTORY_PATH)
    # Se dibuja la curva de accuracy de entrenamiento y validación.
    plot_training_curve(
        history=history.history,
        training_key="accuracy",
        validation_key="val_accuracy",
        title="Curva de accuracy durante el entrenamiento",
        y_label="Accuracy",
        output_path=ACCURACY_FIGURE_PATH,
    )
    # Se dibuja la curva de pérdida de entrenamiento y validación.
    plot_training_curve(
        history=history.history,
        training_key="loss",
        validation_key="val_loss",
        title="Curva de pérdida durante el entrenamiento",
        y_label="Loss",
        output_path=LOSS_FIGURE_PATH,
    )
    # Se guarda una copia del historial también como tabla Markdown-friendly si fuera útil.
    pd.DataFrame(history.history).to_csv(TRAINING_HISTORY_PATH, index=False, encoding="utf-8")
    # Se genera una imagen de ejemplo a partir del dataset para la demostración final.
    sample_info = create_sample_user_image(dataset.test_images, dataset.test_labels)
    # Se ejecuta la evaluación completa sobre el conjunto de prueba.
    evaluation_metrics = evaluate_saved_model(MODEL_PATH)
    # Se ejecuta la predicción sobre la imagen de muestra generada.
    sample_prediction = predict_image(Path(sample_info["ruta_imagen"]), MODEL_PATH)
    # Se combina la verdad terreno con la predicción para dejar evidencia registrada.
    sample_record = {
        "imagen": sample_info["ruta_imagen"],
        "clase_real": sample_info["clase_real"],
        "indice_real": sample_info["indice_real"],
        "prediccion": sample_prediction,
    }
    # Se guarda el registro de demostración de la predicción.
    save_json(SAMPLE_PREDICTION_PATH, sample_record)
    # Se devuelve un resumen integral de la ejecución.
    return {
        "model_path": str(MODEL_PATH),
        "history_path": str(TRAINING_HISTORY_PATH),
        "metrics_path": str(EVALUATION_METRICS_PATH),
        "sample_prediction_path": str(SAMPLE_PREDICTION_PATH),
        "test_accuracy": evaluation_metrics["accuracy"],
        "sample_prediction": sample_prediction,
        "x_test_shape": tuple(x_test.shape),
    }


# Esta función aplica data augmentation y preprocess_input sobre un dataset de TensorFlow.
def _prepare_transfer_dataset(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
    """Añadir aumentos recomendados y normalización oficial al pipeline del dataset propio."""

    # Se construye el bloque de augmentation reutilizable.
    augmentation = build_clothing_augmentation()
    # Se define la función de transformación aplicada lote a lote.
    def transform(images, labels):
        """Transformar imágenes según si el flujo es de entrenamiento o validación."""

        # Se convierten a float32 para operar con capas de Keras de forma estable.
        float_images = tf.cast(images, tf.float32)
        # Si es entrenamiento, se aplican aumentos realistas para prendas.
        augmented_images = augmentation(float_images, training=True) if training else float_images
        # Se normaliza con el preprocesado oficial de MobileNetV2.
        processed_images = tf.keras.applications.mobilenet_v2.preprocess_input(augmented_images)
        # Se devuelven imágenes y etiquetas preservando el contrato del dataset.
        return processed_images, labels

    # Se paraleliza la transformación para mejorar el rendimiento.
    return dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# Esta función concatena dos historiales de entrenamiento conservando el orden temporal.
def _merge_histories(history_phase_1: dict[str, list[float]], history_phase_2: dict[str, list[float]]) -> dict[str, list[float]]:
    """Unir los historiales de ambas fases para graficarlos como una sola secuencia."""

    # Se obtienen todas las claves presentes en ambos historiales.
    all_keys = sorted(set(history_phase_1.keys()) | set(history_phase_2.keys()))
    # Se crea el diccionario final concatenando por clave.
    merged_history = {
        key: list(history_phase_1.get(key, [])) + list(history_phase_2.get(key, []))
        for key in all_keys
    }
    # Se devuelve el historial combinado.
    return merged_history


# Esta función construye los callbacks de regularización para el entrenamiento profesional.
def _build_transfer_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Crear EarlyStopping y ReduceLROnPlateau con parámetros recomendados."""

    # Se devuelve la lista de callbacks requerida por el flujo en dos fases.
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=1,
        ),
    ]


# Esta función entrena el modelo de transferencia en dos fases sin romper el flujo original.
def train_custom_model(data_dir: str | Path = CUSTOM_DATASET_DIR) -> dict:
    """Entrenar MobileNetV2 con dataset propio en dos fases y guardar artefactos."""

    # Se fijan semillas para hacer más estable la reproducibilidad.
    set_global_seed(RANDOM_SEED)
    # Se aseguran las carpetas de salida del proyecto.
    ensure_project_directories(
        [
            TRAINED_MODELS_DIR,
            REPORTS_DIR,
            FIGURES_DIR,
            TABLES_DIR,
        ]
    )
    # Se cargan los datasets del usuario y sus metadatos.
    datasets = load_custom_image_datasets(data_dir)
    # Se preparan las canalizaciones de entrenamiento y validación.
    train_dataset = _prepare_transfer_dataset(datasets.train_dataset, training=True)
    validation_dataset = _prepare_transfer_dataset(datasets.validation_dataset, training=False)
    # Se crea el modelo y la base convolucional preentrenada.
    model, base_model = build_transfer_model(datasets.num_classes)
    # Se compila la fase 1 entrenando sólo la cabeza final.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRANSFER_PHASE_1_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Se entrena la cabeza con la base congelada.
    history_phase_1 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=TRANSFER_PHASE_1_EPOCHS,
        class_weight=datasets.class_weight,
        callbacks=_build_transfer_callbacks(),
        verbose=1,
    )
    # Se descongelan sólo las últimas capas de la base para fine-tuning.
    base_model.trainable = True
    for layer in base_model.layers[:-TRANSFER_FINE_TUNE_LAYERS]:
        layer.trainable = False
    # Se recompila con una tasa de aprendizaje mucho menor.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRANSFER_PHASE_2_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Se continúa el entrenamiento profundo de ajuste fino.
    history_phase_2 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=len(history_phase_1.history.get("loss", [])),
        epochs=len(history_phase_1.history.get("loss", [])) + TRANSFER_PHASE_2_EPOCHS,
        class_weight=datasets.class_weight,
        callbacks=_build_transfer_callbacks(),
        verbose=1,
    )
    # Se combinan ambos historiales para persistencia y visualización.
    merged_history = _merge_histories(history_phase_1.history, history_phase_2.history)
    # Se guarda el modelo de transferencia entrenado.
    model.save(CUSTOM_MODEL_PATH)
    # Se guardan los nombres de clase reales del dataset del usuario.
    save_json(CUSTOM_CLASS_NAMES_PATH, datasets.class_names)
    # Se guarda el historial consolidado.
    save_training_history_csv(merged_history, CUSTOM_TRAINING_HISTORY_PATH)
    pd.DataFrame(merged_history).to_csv(CUSTOM_TRAINING_HISTORY_PATH, index=False, encoding="utf-8")
    # Se dibujan las curvas de accuracy y pérdida del entrenamiento profesional.
    plot_training_curve(
        history=merged_history,
        training_key="accuracy",
        validation_key="val_accuracy",
        title="Curva de accuracy del modelo de transferencia",
        y_label="Accuracy",
        output_path=CUSTOM_ACCURACY_FIGURE_PATH,
    )
    plot_training_curve(
        history=merged_history,
        training_key="loss",
        validation_key="val_loss",
        title="Curva de pérdida del modelo de transferencia",
        y_label="Loss",
        output_path=CUSTOM_LOSS_FIGURE_PATH,
    )
    # Se evalúa el modelo entrenado para generar métricas y figuras asociadas.
    metrics = evaluate_custom_model(CUSTOM_MODEL_PATH, data_dir=data_dir)
    # Se resume la estructura del dataset para trazabilidad.
    dataset_summary = summarize_dataset_structure(data_dir)
    # Se devuelve un resumen completo del proceso.
    return {
        "model_path": str(CUSTOM_MODEL_PATH),
        "history_path": str(CUSTOM_TRAINING_HISTORY_PATH),
        "metrics_path": str(CUSTOM_EVALUATION_METRICS_PATH),
        "data_dir": str(Path(data_dir)),
        "validation_split": CUSTOM_VALIDATION_SPLIT,
        "class_names": datasets.class_names,
        "dataset_summary": dataset_summary,
        "class_weight": datasets.class_weight,
        "accuracy": metrics["accuracy"],
        "roc_auc_macro_ovr": metrics["roc_auc_macro_ovr"],
    }