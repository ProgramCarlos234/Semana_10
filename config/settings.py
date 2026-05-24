from __future__ import annotations

# Se importa Path para definir todas las rutas del proyecto de manera portable.
from pathlib import Path


# Se calcula la raíz absoluta del proyecto a partir de este archivo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Se define la carpeta principal de datos.
DATA_DIR = PROJECT_ROOT / "data"
# Se define la carpeta de datos originales o descargados.
RAW_DATA_DIR = DATA_DIR / "raw"
# Se define la carpeta de datos transformados.
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# Se define la carpeta para imágenes de ejemplo aportadas por el usuario o generadas.
SAMPLE_USER_IMAGES_DIR = DATA_DIR / "sample_user_images"

# Se define la carpeta principal de modelos.
MODELS_DIR = PROJECT_ROOT / "models"
# Se define la carpeta específica para modelos entrenados.
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
# Se define la carpeta de informes numéricos y registros.
REPORTS_DIR = MODELS_DIR / "reports"

# Se define la carpeta principal de documentación.
DOCS_DIR = PROJECT_ROOT / "docs"
# Se define la carpeta de figuras del proyecto.
FIGURES_DIR = DOCS_DIR / "figuras"
# Se define la carpeta de tablas del proyecto.
TABLES_DIR = DOCS_DIR / "tablas"

# Se define la ruta del modelo entrenado final.
MODEL_PATH = TRAINED_MODELS_DIR / "fashion_cnn.keras"
# Se define la ruta del modelo de transferencia entrenado con dataset propio.
CUSTOM_MODEL_PATH = TRAINED_MODELS_DIR / "fashion_transfer.keras"
# Se define la ruta del archivo JSON con nombres de clases.
CLASS_NAMES_PATH = TRAINED_MODELS_DIR / "class_names.json"
# Se define la ruta del archivo JSON con nombres de clases del dataset propio.
CUSTOM_CLASS_NAMES_PATH = TRAINED_MODELS_DIR / "class_names_custom.json"
# Se define la ruta del historial del entrenamiento en CSV.
TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history.csv"
# Se define la ruta del historial del entrenamiento con transferencia.
CUSTOM_TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history_transfer.csv"
# Se define la ruta de la matriz de confusión tabular.
CONFUSION_MATRIX_CSV_PATH = REPORTS_DIR / "confusion_matrix.csv"
# Se define la ruta de la matriz de confusión del modelo de transferencia.
CUSTOM_CONFUSION_MATRIX_CSV_PATH = REPORTS_DIR / "confusion_matrix_transfer.csv"
# Se define la ruta de las métricas agregadas de evaluación.
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
# Se define la ruta de las métricas agregadas del modelo de transferencia.
CUSTOM_EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics_transfer.json"
# Se define la ruta del registro de predicción de la imagen de ejemplo.
SAMPLE_PREDICTION_PATH = REPORTS_DIR / "sample_prediction.json"

# Se define la ruta de la figura de accuracy.
ACCURACY_FIGURE_PATH = FIGURES_DIR / "curva_accuracy.png"
# Se define la ruta de la figura de pérdida.
LOSS_FIGURE_PATH = FIGURES_DIR / "curva_loss.png"
# Se define la ruta de la figura visual de la matriz de confusión.
CONFUSION_MATRIX_FIGURE_PATH = FIGURES_DIR / "matriz_confusion.png"
# Se define la ruta de la figura con ejemplos clasificados.
CLASSIFIED_EXAMPLES_FIGURE_PATH = FIGURES_DIR / "ejemplos_clasificados.png"
# Se define la ruta de la figura de accuracy del modo transferencia.
CUSTOM_ACCURACY_FIGURE_PATH = FIGURES_DIR / "curva_accuracy_transfer.png"
# Se define la ruta de la figura de pérdida del modo transferencia.
CUSTOM_LOSS_FIGURE_PATH = FIGURES_DIR / "curva_loss_transfer.png"
# Se define la ruta de la figura de matriz de confusión del modo transferencia.
CUSTOM_CONFUSION_MATRIX_FIGURE_PATH = FIGURES_DIR / "matriz_confusion_transfer.png"
# Se define la ruta de la figura de ejemplos clasificados del modo transferencia.
CUSTOM_CLASSIFIED_EXAMPLES_FIGURE_PATH = FIGURES_DIR / "ejemplos_clasificados_transfer.png"
# Se define la ruta de la curva ROC multiclase del modo transferencia.
CUSTOM_ROC_AUC_FIGURE_PATH = FIGURES_DIR / "roc_auc_transfer.png"

# Se define la ruta de la imagen sintética de ejemplo para demostración.
SAMPLE_IMAGE_PATH = SAMPLE_USER_IMAGES_DIR / "imagen_ejemplo_dataset.png"

# Se define la carpeta del dataset propio siguiendo una estructura tipo ImageFolder.
CUSTOM_DATASET_DIR = DATA_DIR / "custom_dataset"

# Se fija el alto de entrada de la imagen.
IMAGE_HEIGHT = 28
# Se fija el ancho de entrada de la imagen.
IMAGE_WIDTH = 28
# Se fija el número de canales del problema.
IMAGE_CHANNELS = 1
# Se declara la forma de entrada esperada por la CNN.
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
# Se fija el alto de entrada del modo transferencia.
TRANSFER_IMAGE_HEIGHT = 224
# Se fija el ancho de entrada del modo transferencia.
TRANSFER_IMAGE_WIDTH = 224
# Se fija el número de canales RGB del modo transferencia.
TRANSFER_IMAGE_CHANNELS = 3
# Se declara la forma de entrada esperada por MobileNetV2.
TRANSFER_INPUT_SHAPE = (TRANSFER_IMAGE_HEIGHT, TRANSFER_IMAGE_WIDTH, TRANSFER_IMAGE_CHANNELS)

# Se define el número de clases del problema Fashion MNIST.
NUM_CLASSES = 10
# Se define el tamaño de lote exigido por el plan.
BATCH_SIZE = 32
# Se define el número de épocas exigido por el plan.
EPOCHS = 10
# Se define la fracción de validación interna usada durante el entrenamiento.
VALIDATION_SPLIT = 0.1
# Se define la semilla global para reproducibilidad razonable.
RANDOM_SEED = 42
# Se define la fracción de validación para el dataset propio.
CUSTOM_VALIDATION_SPLIT = 0.2
# Se define el número de épocas recomendado para la fase 1 del modo transferencia.
TRANSFER_PHASE_1_EPOCHS = 30
# Se define el número de épocas recomendado para la fase 2 del modo transferencia.
TRANSFER_PHASE_2_EPOCHS = 15
# Se define la tasa de aprendizaje de la fase 1.
TRANSFER_PHASE_1_LEARNING_RATE = 1e-3
# Se define la tasa de aprendizaje de la fase 2.
TRANSFER_PHASE_2_LEARNING_RATE = 1e-5
# Se define el dropout recomendado para la cabeza del clasificador.
TRANSFER_DROPOUT_RATE = 0.4
# Se define la regularización L2 de la capa final.
TRANSFER_L2_FACTOR = 1e-4
# Se define el número de capas finales a descongelar en el fine-tuning.
TRANSFER_FINE_TUNE_LAYERS = 30
# Se define la paciencia del early stopping.
EARLY_STOPPING_PATIENCE = 5
# Se define la paciencia del ajuste dinámico de la tasa de aprendizaje.
REDUCE_LR_PATIENCE = 3
# Se define el factor de reducción de learning rate.
REDUCE_LR_FACTOR = 0.2

# Se define el nombre del optimizador exigido por el plan.
OPTIMIZER_NAME = "adam"
# Se define la función de pérdida exigida por el plan.
LOSS_NAME = "sparse_categorical_crossentropy"
# Se define la lista de métricas del entrenamiento.
METRICS = ["accuracy"]

# Se define el título de la ventana principal de la interfaz.
GUI_TITLE = "Clasificador inteligente de prendas con CNN"
# Se define el tamaño inicial de la ventana de la aplicación.
GUI_WINDOW_SIZE = "960x760"
# Se define el tamaño máximo de la vista previa.
GUI_PREVIEW_SIZE = (260, 260)

# Se define la lista oficial de nombres de clases en español de España.
CLASS_NAMES = [
    "Camiseta o top",
    "Pantalón",
    "Jersey",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla deportiva",
    "Short",
    "Botín",
]