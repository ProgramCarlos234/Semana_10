from __future__ import annotations

# Se importa tkinter como base de la interfaz gráfica.
import tkinter as tk
# Se importa Path para almacenar la ruta de la imagen seleccionada.
from pathlib import Path
# Se importan diálogos y widgets avanzados de Tkinter.
from tkinter import filedialog, messagebox, ttk

# Se importan constantes de configuración de la GUI.
from config.settings import CUSTOM_MODEL_PATH, GUI_TITLE, GUI_WINDOW_SIZE, MODEL_PATH
# Se importa la función que construye la vista previa de la imagen.
from src.gui.image_handlers import build_preview_image
# Se importan widgets reutilizables del proyecto.
from src.gui.widgets import create_body_label, create_button, create_frame, create_title
# Se importa la predicción del modelo sobre imágenes externas.
from src.predictor import predict_image


# Esta clase encapsula la interfaz gráfica completa del proyecto.
class FashionClassifierApp:
    """Gestionar la ventana, los eventos y la interacción del usuario con el modelo."""

    # Este método construye todos los componentes visuales y el estado inicial.
    def __init__(self, root: tk.Tk) -> None:
        """Inicializar la ventana principal y sus controles."""

        # Se guarda la referencia a la raíz de Tkinter.
        self.root = root
        # Se fija el título principal de la aplicación.
        self.root.title(GUI_TITLE)
        # Se fija el tamaño inicial de la ventana.
        self.root.geometry(GUI_WINDOW_SIZE)
        # Se establece un tamaño mínimo razonable para evitar solapamientos severos.
        self.root.minsize(900, 700)

        # Se inicializa la ruta de imagen seleccionada como inexistente.
        self.selected_image_path: Path | None = None
        # Se inicializa la referencia a la imagen Tk para evitar que el recolector la elimine.
        self.preview_image = None
        # Se decide qué modelo debe usar la interfaz priorizando el modelo profesional.
        self.active_model_path = self._resolve_model_path()

        # Se crea el contenedor principal de toda la interfaz.
        self.main_frame = create_frame(self.root, padding=18)
        # Se coloca el contenedor principal ocupando toda la ventana.
        self.main_frame.pack(fill="both", expand=True)

        # Se crea el título principal del proyecto.
        self.title_label = create_title(self.main_frame, "Sistema inteligente de clasificación de prendas")
        # Se coloca el título con margen inferior.
        self.title_label.pack(pady=(0, 12))

        # Se crea un texto descriptivo del flujo de uso de la aplicación.
        self.description_label = create_body_label(
            self.main_frame,
            "Carga una imagen de una prenda, visualiza la previsualización y pulsa en predecir para obtener la clase estimada y las tres probabilidades más altas.",
        )
        # Se coloca el texto descriptivo en la ventana.
        self.description_label.pack(pady=(0, 16))

        # Se crea un marco para los botones de acción.
        self.button_frame = create_frame(self.main_frame, padding=0)
        # Se coloca el marco de botones.
        self.button_frame.pack(pady=(0, 16))

        # Se crea el botón para seleccionar una imagen desde disco.
        self.load_button = create_button(self.button_frame, "Cargar imagen", self.select_image)
        # Se sitúa el botón en la rejilla interna.
        self.load_button.grid(row=0, column=0, padx=8)

        # Se crea el botón para ejecutar la predicción.
        self.predict_button = create_button(self.button_frame, "Predecir", self.run_prediction)
        # Se sitúa el botón a la derecha del anterior.
        self.predict_button.grid(row=0, column=1, padx=8)

        # Se crea el botón para limpiar la imagen y los resultados actuales.
        self.clear_button = create_button(self.button_frame, "Limpiar", self.clear_selection)
        # Se sitúa el botón a la derecha del flujo principal.
        self.clear_button.grid(row=0, column=2, padx=8)

        # Se crea una etiqueta informativa sobre el estado del modelo.
        self.model_status_label = create_body_label(
            self.main_frame,
            self._build_model_status_text(),
        )
        # Se coloca la etiqueta de estado.
        self.model_status_label.pack(pady=(0, 10))

        # Se crea una etiqueta que mostrará la ruta o el nombre de la imagen seleccionada.
        self.file_label = create_body_label(self.main_frame, "Imagen seleccionada: ninguna")
        # Se coloca la etiqueta del fichero seleccionada.
        self.file_label.pack(pady=(0, 12))

        # Se crea la etiqueta vacía donde aparecerá la vista previa.
        self.preview_label = ttk.Label(self.main_frame)
        # Se coloca la vista previa en la interfaz.
        self.preview_label.pack(pady=(0, 16))

        # Se crea una etiqueta donde se mostrará la predicción principal.
        self.result_label = create_body_label(self.main_frame, "Resultado: pendiente")
        # Se coloca la etiqueta de resultado.
        self.result_label.pack(pady=(0, 12))

        # Se crea una etiqueta de estado para informar del procesamiento en curso.
        self.loading_label = create_body_label(self.main_frame, "Estado: en espera")
        # Se coloca la etiqueta de estado operativo.
        self.loading_label.pack(pady=(0, 12))

        # Se crea una etiqueta para introducir la sección de top-3 probabilidades.
        self.top_label = create_body_label(self.main_frame, "Top-3 probabilidades con confianza:")
        # Se coloca la etiqueta correspondiente.
        self.top_label.pack(anchor="w")

        # Se crea un contenedor específico para las barras de confianza.
        self.top3_frame = create_frame(self.main_frame, padding=0)
        # Se coloca el bloque de barras en la ventana principal.
        self.top3_frame.pack(fill="x", pady=(6, 12))

        # Se inicializan las filas visuales del top-3.
        self.top3_rows: list[dict[str, ttk.Widget]] = []
        # Se crean exactamente tres filas para mantener la interfaz estable.
        for row_index in range(3):
            # Se crea una fila contenedora por cada predicción mostrada.
            row_frame = create_frame(self.top3_frame, padding=0)
            # Se coloca la fila ocupando todo el ancho disponible.
            row_frame.pack(fill="x", pady=4)
            # Se crea la etiqueta de nombre de clase.
            class_label = ttk.Label(row_frame, text=f"{row_index + 1}. ---", font=("Segoe UI", 10))
            # Se alinea la clase a la izquierda con ancho fijo.
            class_label.pack(side="left", padx=(0, 10))
            # Se crea la barra de progreso horizontal.
            progress = ttk.Progressbar(row_frame, orient="horizontal", mode="determinate", maximum=100, length=340)
            # Se coloca la barra en el centro expandible.
            progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
            # Se crea la etiqueta del porcentaje numérico.
            percent_label = ttk.Label(row_frame, text="0.00 %", font=("Segoe UI", 10, "bold"))
            # Se coloca el porcentaje a la derecha.
            percent_label.pack(side="right")
            # Se registra la fila para su actualización posterior.
            self.top3_rows.append(
                {
                    "class_label": class_label,
                    "progress": progress,
                    "percent_label": percent_label,
                }
            )

        # Se deja la interfaz en estado limpio inicial.
        self._reset_top3_rows()

    # Este método resuelve qué modelo debe usar la GUI según los artefactos disponibles.
    def _resolve_model_path(self) -> Path:
        """Seleccionar el modelo profesional si existe y, si no, mantener compatibilidad."""

        # Si ya existe el modelo de transferencia, se prioriza por ser más robusto con imágenes reales.
        if CUSTOM_MODEL_PATH.exists():
            return CUSTOM_MODEL_PATH
        # En caso contrario se conserva el comportamiento clásico del proyecto.
        return MODEL_PATH

    # Este método construye un texto informativo legible sobre el modelo activo.
    def _build_model_status_text(self) -> str:
        """Crear el mensaje que informa del modelo actualmente disponible en la interfaz."""

        # Si existe el modelo profesional, se comunica como opción prioritaria.
        if CUSTOM_MODEL_PATH.exists():
            return "Estado del modelo: usando transfer learning para imágenes reales."
        # Si sólo existe el modelo simple, se informa de la compatibilidad disponible.
        if MODEL_PATH.exists():
            return "Estado del modelo: usando la CNN simple de Fashion MNIST por compatibilidad."
        # Si no existe ninguno, se guía al usuario con los comandos adecuados.
        return "Estado del modelo: no hay modelos entrenados. Ejecuta `python main.py train` o `python main.py train_custom --data-dir data/custom_dataset`."

    # Este método permite al usuario seleccionar una imagen desde el explorador.
    def select_image(self) -> None:
        """Abrir un diálogo de selección de imagen y actualizar la vista previa."""

        # Se abre el diálogo de selección con filtros de imagen habituales.
        selected_path = filedialog.askopenfilename(
            title="Selecciona una imagen de una prenda",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        # Si el usuario cancela la selección, se finaliza sin cambios.
        if not selected_path:
            return
        # Se guarda la ruta elegida como objeto Path.
        self.selected_image_path = Path(selected_path)
        # Se intenta construir la vista previa y actualizar el estado visual.
        try:
            # Se construye una vista previa compatible con Tkinter.
            self.preview_image = build_preview_image(self.selected_image_path)
            # Se actualiza la etiqueta de previsualización.
            self.preview_label.configure(image=self.preview_image)
        # Si la imagen no puede procesarse, se limpia la selección y se informa con claridad.
        except Exception as error:
            self.clear_selection()
            messagebox.showerror("Imagen no válida", f"No se pudo abrir la imagen seleccionada: {error}")
            return
        # Se muestra el nombre del archivo cargado.
        self.file_label.configure(text=f"Imagen seleccionada: {self.selected_image_path}")
        # Se reinicia el área de resultados para reflejar que hay una nueva imagen pendiente.
        self.result_label.configure(text="Resultado: imagen lista para predecir")
        # Se actualiza el indicador de estado para guiar al usuario.
        self.loading_label.configure(text="Estado: imagen cargada correctamente")
        # Se reinicia el top-3 visual.
        self._reset_top3_rows()

    # Este método lanza la inferencia y actualiza todos los elementos de salida.
    def run_prediction(self) -> None:
        """Predecir la imagen seleccionada y mostrar resultados manejando errores."""

        # Si aún no se ha seleccionado ninguna imagen, se avisa al usuario.
        if self.selected_image_path is None:
            messagebox.showwarning("Falta una imagen", "Primero debes cargar una imagen válida.")
            return
        # Se intenta ejecutar la inferencia sobre la imagen seleccionada.
        try:
            # Se actualiza el estado visual para reflejar el procesamiento.
            self.loading_label.configure(text="Estado: procesando imagen, por favor espera...")
            # Se fuerza el refresco de la GUI antes de la inferencia.
            self.root.update_idletasks()
            # Se llama al módulo de predicción principal del proyecto.
            prediction = predict_image(self.selected_image_path, model_path=self.active_model_path)
        # Si el modelo no existe, se informa con un mensaje guiado.
        except FileNotFoundError:
            messagebox.showerror(
                "Modelo no encontrado",
                "No se ha encontrado ningún modelo utilizable. Ejecuta `python main.py train` o `python main.py train_custom --data-dir data/custom_dataset`.",
            )
            self.loading_label.configure(text="Estado: error por ausencia de modelo")
            return
        # Si ocurre cualquier otro error, se muestra el detalle para facilitar el diagnóstico.
        except Exception as error:
            messagebox.showerror("Error de predicción", f"No se pudo procesar la imagen: {error}")
            self.loading_label.configure(text="Estado: error durante la predicción")
            return
        # Se actualiza el resultado principal con la clase más probable.
        self.result_label.configure(
            text=f"Resultado: {prediction['clase_predicha']} ({prediction['probabilidad'] * 100:.2f} %)"
        )
        # Se vuelcan las tres probabilidades principales en barras horizontales.
        self._update_top3_rows(prediction["top_3"])
        # Se informa del modelo realmente empleado para la inferencia.
        self.loading_label.configure(
            text=f"Estado: predicción completada con el modelo {prediction['modelo_utilizado']}"
        )

    # Este método permite limpiar la imagen actual y devolver la interfaz a estado neutro.
    def clear_selection(self) -> None:
        """Limpiar la selección de imagen, la vista previa y los resultados mostrados."""

        # Se elimina la referencia a la imagen seleccionada.
        self.selected_image_path = None
        # Se elimina la referencia Tk para que no quede una previsualización obsoleta.
        self.preview_image = None
        # Se limpia la vista previa visual.
        self.preview_label.configure(image="")
        # Se restaura el texto base de la ruta seleccionada.
        self.file_label.configure(text="Imagen seleccionada: ninguna")
        # Se restaura el estado de resultado.
        self.result_label.configure(text="Resultado: pendiente")
        # Se restaura el indicador de procesamiento.
        self.loading_label.configure(text="Estado: en espera")
        # Se vacían las barras de confianza.
        self._reset_top3_rows()

    # Este método actualiza las barras horizontales del top-3.
    def _update_top3_rows(self, top_predictions: list[dict]) -> None:
        """Representar las tres clases más probables mediante etiquetas y barras."""

        # Se recorren las filas visuales y las predicciones disponibles en paralelo.
        for row_index, row_widgets in enumerate(self.top3_rows):
            # Si existe una predicción para esta fila, se actualizan sus valores.
            if row_index < len(top_predictions):
                prediction = top_predictions[row_index]
                probability_percent = prediction["probabilidad"] * 100.0
                row_widgets["class_label"].configure(text=f"{row_index + 1}. {prediction['clase']}")
                row_widgets["progress"]["value"] = probability_percent
                row_widgets["percent_label"].configure(text=f"{probability_percent:.2f} %")
            # Si no hay dato, se limpia la fila correspondiente.
            else:
                row_widgets["class_label"].configure(text=f"{row_index + 1}. ---")
                row_widgets["progress"]["value"] = 0.0
                row_widgets["percent_label"].configure(text="0.00 %")

    # Este método restablece el top-3 cuando no hay predicción activa.
    def _reset_top3_rows(self) -> None:
        """Vaciar la sección visual del top-3 dejando una interfaz limpia y coherente."""

        # Se reinicia cada fila con su estado neutro.
        for row_index, row_widgets in enumerate(self.top3_rows):
            row_widgets["class_label"].configure(text=f"{row_index + 1}. Pendiente")
            row_widgets["progress"]["value"] = 0.0
            row_widgets["percent_label"].configure(text="0.00 %")


# Esta función construye la aplicación y entra en el bucle principal de eventos.
def launch_gui() -> None:
    """Crear la ventana raíz y lanzar la aplicación Tkinter del proyecto."""

    # Se crea la ventana raíz de Tkinter.
    root = tk.Tk()
    # Se instancia la aplicación completa sobre la raíz recién creada.
    FashionClassifierApp(root)
    # Se inicia el bucle de eventos de la interfaz gráfica.
    root.mainloop()
