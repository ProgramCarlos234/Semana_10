from __future__ import annotations

# Se importa Path para manejar rutas de salida.
from pathlib import Path

# Se importa matplotlib para generar gráficos académicos.
import matplotlib.pyplot as plt
# Se importa NumPy para manipular matrices en la visualización.
import numpy as np
# Se importa pandas para exportar tablas CSV del historial.
import pandas as pd

# Se importan nombres de clases para etiquetar la matriz de confusión.
from config.settings import CLASS_NAMES
# Se importa la utilidad para asegurar carpetas de salida.
from src.utils import ensure_directory


# Esta función vuelca el historial de entrenamiento a un CSV.
def save_training_history_csv(history: dict[str, list[float]], output_path: Path) -> None:
    """Guardar el historial de entrenamiento en un CSV reutilizable para informes."""

    # Se garantiza que la carpeta de salida exista.
    ensure_directory(output_path.parent)
    # Se convierte el historial de Keras en un DataFrame tabular.
    history_frame = pd.DataFrame(history)
    # Se guarda el DataFrame en formato CSV sin índice.
    history_frame.to_csv(output_path, index=False, encoding="utf-8")


# Esta función dibuja una curva de entrenamiento frente a validación.
def plot_training_curve(
    history: dict[str, list[float]],
    training_key: str,
    validation_key: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Representar en PNG la evolución de una métrica durante el entrenamiento."""

    # Se garantiza que exista la carpeta donde se guardará la figura.
    ensure_directory(output_path.parent)
    # Se crea una nueva figura con un tamaño apto para la memoria académica.
    plt.figure(figsize=(9, 5))
    # Se dibuja la serie de entrenamiento.
    plt.plot(history[training_key], marker="o", label="Entrenamiento")
    # Se dibuja la serie de validación.
    plt.plot(history[validation_key], marker="s", label="Validación")
    # Se asigna un título descriptivo.
    plt.title(title)
    # Se etiqueta el eje horizontal.
    plt.xlabel("Época")
    # Se etiqueta el eje vertical.
    plt.ylabel(y_label)
    # Se añade una rejilla suave para facilitar la lectura.
    plt.grid(alpha=0.3)
    # Se añade la leyenda explicativa.
    plt.legend()
    # Se ajusta el diseño para evitar solapamientos.
    plt.tight_layout()
    # Se guarda la figura en la ruta indicada.
    plt.savefig(output_path, dpi=180)
    # Se cierra la figura para liberar memoria.
    plt.close()


# Esta función representa visualmente una matriz de confusión.
def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    class_names: list[str] | None = None,
    title: str = "Matriz de confusión del modelo CNN",
) -> None:
    """Guardar una matriz de confusión visual con etiquetas en español."""

    # Si no se recibe un conjunto de clases explícito, se usan las clases por defecto.
    labels = class_names or CLASS_NAMES
    # Se garantiza que la carpeta de salida exista antes de dibujar.
    ensure_directory(output_path.parent)
    # Se crea una figura cuadrada amplia para mostrar bien los diez rótulos.
    plt.figure(figsize=(10, 8))
    # Se dibuja la matriz con una paleta de azules.
    plt.imshow(matrix, cmap="Blues")
    # Se añade una barra de color para interpretar la intensidad.
    plt.colorbar()
    # Se fija el título de la figura.
    plt.title(title)
    # Se colocan las etiquetas del eje X con rotación para mejorar la legibilidad.
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    # Se colocan las etiquetas del eje Y.
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    # Se etiqueta el eje horizontal.
    plt.xlabel("Clase predicha")
    # Se etiqueta el eje vertical.
    plt.ylabel("Clase real")
    # Se recorre cada celda para escribir su valor numérico.
    for row_index in range(matrix.shape[0]):
        # Se recorre cada columna de la fila actual.
        for column_index in range(matrix.shape[1]):
            # Se escribe el valor de la celda centrado sobre la imagen.
            plt.text(
                column_index,
                row_index,
                str(int(matrix[row_index, column_index])),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    # Se ajusta el diseño final de la figura.
    plt.tight_layout()
    # Se guarda la matriz de confusión visual.
    plt.savefig(output_path, dpi=180)
    # Se cierra la figura para liberar recursos.
    plt.close()


# Esta función genera una figura con ejemplos correctamente clasificados.
def plot_classified_examples(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    class_names: list[str] | None = None,
    title: str = "Ejemplos correctamente clasificados por la CNN",
    cmap: str | None = "gray",
) -> None:
    """Guardar un mosaico de ejemplos clasificados para la exposición del proyecto."""

    # Si no se reciben clases explícitas, se usan las del problema base.
    labels_text = class_names or CLASS_NAMES
    # Se garantiza la existencia del directorio de salida.
    ensure_directory(output_path.parent)
    # Se localizan los índices de aciertos para mostrar ejemplos representativos.
    correct_indexes = np.where(labels == predictions)[0][:9]
    # Se crea una figura con una rejilla de 3x3.
    plt.figure(figsize=(10, 10))
    # Se itera sobre los índices seleccionados para llenar la rejilla.
    for plot_position, sample_index in enumerate(correct_indexes, start=1):
        # Se crea un subgráfico para la imagen actual.
        plt.subplot(3, 3, plot_position)
        # Se muestra la imagen en escala de grises.
        plt.imshow(images[sample_index], cmap=cmap)
        # Se escribe un título con clase real y clase predicha.
        plt.title(
            f"Real: {labels_text[int(labels[sample_index])]}\nPred.: {labels_text[int(predictions[sample_index])]}",
            fontsize=9,
        )
        # Se ocultan los ejes para centrar la atención en la prenda.
        plt.axis("off")
    # Se añade un título general a la figura.
    plt.suptitle(title, fontsize=14)
    # Se ajusta el diseño dejando espacio para el título general.
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    # Se guarda la figura final.
    plt.savefig(output_path, dpi=180)
    # Se cierra la figura abierta.
    plt.close()


# Esta función representa curvas ROC multiclase para la validación del modo profesional.
def plot_multiclass_roc_auc(roc_curves: list[dict[str, object]], output_path: Path) -> None:
    """Guardar una figura ROC multiclase con una curva por categoría."""

    # Se garantiza la existencia del directorio de salida.
    ensure_directory(output_path.parent)
    # Se crea una figura amplia para acomodar varias clases.
    plt.figure(figsize=(10, 8))
    # Se dibuja la diagonal de referencia de un clasificador aleatorio.
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    # Se recorren las curvas calculadas para cada clase.
    for curve in roc_curves:
        # Se representa la curva con su etiqueta correspondiente.
        plt.plot(curve["fpr"], curve["tpr"], label=str(curve["class_name"]))
    # Se completa la rotulación académica de la figura.
    plt.title("Curvas ROC-AUC multiclase")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    # Se guarda la figura en disco.
    plt.savefig(output_path, dpi=180)
    # Se cierra la figura para liberar recursos.
    plt.close()