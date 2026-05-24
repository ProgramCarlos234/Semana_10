from __future__ import annotations

# Se importa importlib para detectar dependencias opcionales en tiempo de ejecución.
import importlib
# Se importa io para reconstruir imágenes devueltas por dependencias opcionales.
import io
# Se importa Path para aceptar rutas flexibles.
from pathlib import Path

# Se importa NumPy para manipular matrices de imagen.
import numpy as np
# Se importa Pillow para la carga básica de imágenes.
from PIL import Image, ImageOps
# Se importa preprocess_input de MobileNetV2 para normalización correcta.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Se importan constantes de resolución del modo transferencia.
from config.settings import TRANSFER_IMAGE_HEIGHT, TRANSFER_IMAGE_WIDTH


# Esta función comprueba si OpenCV está instalado.
def _is_opencv_available() -> bool:
    """Indicar si OpenCV puede utilizarse para mejora de imagen y segmentación."""

    # Se intenta localizar el módulo sin importarlo de forma rígida.
    return importlib.util.find_spec("cv2") is not None


# Esta función comprueba si rembg está instalado.
def _is_rembg_available() -> bool:
    """Indicar si rembg está disponible para eliminación de fondo."""

    # Se intenta localizar el módulo sin convertirlo en dependencia obligatoria.
    return importlib.util.find_spec("rembg") is not None


# Esta función aplica mejoras suaves de iluminación y ruido con OpenCV.
def _enhance_with_opencv(rgb_array: np.ndarray) -> np.ndarray:
    """Aplicar CLAHE y reducción de ruido cuando OpenCV esté disponible."""

    # Si OpenCV no está disponible, se devuelve la imagen original sin fallar.
    if not _is_opencv_available():
        return rgb_array
    # Se importa OpenCV sólo cuando puede usarse.
    import cv2

    # Se transforma la imagen RGB al espacio LAB para tratar la luminancia.
    lab_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    # Se separan los canales de luminancia y crominancia.
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    # Se crea un CLAHE moderado para no introducir artefactos en la prenda.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Se mejora el contraste local sobre la luminancia.
    enhanced_l_channel = clahe.apply(l_channel)
    # Se recompone la imagen LAB con la luminancia mejorada.
    merged_lab = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    # Se vuelve al espacio RGB para continuar el pipeline.
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    # Se aplica una reducción suave de ruido sin destruir texturas de ropa.
    denoised_rgb = cv2.fastNlMeansDenoisingColored(enhanced_rgb, None, 3, 3, 7, 21)
    # Se devuelve la imagen mejorada.
    return denoised_rgb


# Esta función intenta eliminar el fondo con rembg si está instalado.
def _remove_background_with_rembg(rgb_image: Image.Image) -> Image.Image:
    """Eliminar el fondo mediante rembg/U2Net cuando la dependencia exista."""

    # Si rembg no está disponible, se deja la imagen intacta.
    if not _is_rembg_available():
        return rgb_image
    # Se importa la función de eliminación de fondo sólo de forma opcional.
    from rembg import remove

    # Se intenta primero la API moderna que acepta imágenes PIL.
    removed_background = remove(rgb_image)
    # Si la salida ya es una imagen PIL, se normaliza el modo y se devuelve.
    if isinstance(removed_background, Image.Image):
        return removed_background.convert("RGBA")
    # Si la librería devuelve bytes PNG, se reconstruye una imagen PIL.
    if isinstance(removed_background, bytes):
        return Image.open(io.BytesIO(removed_background)).convert("RGBA")
    # Si devuelve un array u otro objeto compatible, se convierte de forma segura.
    return Image.fromarray(np.array(removed_background)).convert("RGBA")


# Esta función aplica una segmentación de respaldo con GrabCut.
def _remove_background_with_grabcut(rgb_array: np.ndarray) -> np.ndarray:
    """Aplicar una segmentación aproximada con GrabCut cuando OpenCV esté disponible."""

    # Si OpenCV no está instalado, se devuelve la entrada sin modificar.
    if not _is_opencv_available():
        return rgb_array
    # Se importa OpenCV sólo cuando realmente puede ejecutarse.
    import cv2

    # Se crea la máscara inicial requerida por GrabCut.
    mask = np.zeros(rgb_array.shape[:2], np.uint8)
    # Se reservan los modelos internos de fondo y primer plano.
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    # Se define un rectángulo central que aproxima la ubicación de la prenda.
    margin_x = max(int(rgb_array.shape[1] * 0.08), 1)
    margin_y = max(int(rgb_array.shape[0] * 0.08), 1)
    rect = (
        margin_x,
        margin_y,
        max(rgb_array.shape[1] - 2 * margin_x, 1),
        max(rgb_array.shape[0] - 2 * margin_y, 1),
    )
    # Se ejecuta GrabCut con pocas iteraciones para mantenerlo ligero.
    cv2.grabCut(rgb_array, mask, rect, background_model, foreground_model, 3, cv2.GC_INIT_WITH_RECT)
    # Se conserva como primer plano tanto el seguro como el probable.
    foreground_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    # Se aplica la máscara sobre la imagen RGB original.
    segmented_rgb = rgb_array * foreground_mask[:, :, np.newaxis]
    # Se devuelve la imagen segmentada.
    return segmented_rgb


# Esta función centra la prenda sobre un lienzo cuadrado antes de redimensionar.
def _center_on_square_canvas(rgb_image: Image.Image) -> Image.Image:
    """Centrar el contenido útil sobre un fondo neutro para estabilizar la inferencia."""

    # Se calcula el tamaño del lienzo cuadrado a partir del lado mayor.
    canvas_size = max(rgb_image.size)
    # Se crea un fondo blanco uniforme para evitar bordes negros artificiales.
    square_canvas = Image.new("RGB", (canvas_size, canvas_size), color=(255, 255, 255))
    # Se calcula el desplazamiento necesario para centrar la imagen.
    offset_x = (canvas_size - rgb_image.width) // 2
    offset_y = (canvas_size - rgb_image.height) // 2
    # Se pega la imagen en el centro del lienzo.
    square_canvas.paste(rgb_image, (offset_x, offset_y))
    # Se devuelve la composición centrada.
    return square_canvas


# Esta función preprocesa una imagen real para MobileNetV2.
def preprocess_real_image(
    image_path: str | Path,
    remove_background: bool = True,
) -> np.ndarray:
    """Preparar una imagen real con RGB, mejora opcional y preprocess_input."""

    # Se normaliza la ruta recibida.
    input_path = Path(image_path)
    # Si la imagen no existe, se aborta con un mensaje claro.
    if not input_path.exists():
        raise FileNotFoundError(f"No se ha encontrado la imagen indicada: {input_path}")
    # Se abre la imagen y se convierte a RGB para trabajar con redes preentrenadas.
    rgb_image = Image.open(input_path).convert("RGB")
    # Se corrige la orientación EXIF cuando exista.
    rgb_image = ImageOps.exif_transpose(rgb_image)
    # Si se solicita, se intenta una eliminación de fondo robusta pero opcional.
    if remove_background:
        # Primero se prueba rembg si está instalado.
        if _is_rembg_available():
            try:
                # Se aplica rembg y se compone sobre fondo blanco si hay alfa.
                rgba_image = _remove_background_with_rembg(rgb_image)
                white_background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
                rgb_image = Image.alpha_composite(white_background, rgba_image).convert("RGB")
            # Si rembg falla por cualquier motivo, se usa el camino clásico con OpenCV.
            except Exception:
                rgb_image = Image.fromarray(_remove_background_with_grabcut(np.array(rgb_image, dtype=np.uint8)))
        # Si rembg no existe, se utiliza GrabCut como alternativa opcional.
        else:
            rgb_image = Image.fromarray(_remove_background_with_grabcut(np.array(rgb_image, dtype=np.uint8)))
    # Se aplican mejoras suaves de iluminación y reducción de ruido cuando sea posible.
    enhanced_rgb_array = _enhance_with_opencv(np.array(rgb_image, dtype=np.uint8))
    # Se reconstruye la imagen PIL para continuar con operaciones geométricas.
    enhanced_rgb_image = Image.fromarray(enhanced_rgb_array)
    # Se centra la prenda sobre un lienzo cuadrado antes del redimensionado final.
    centered_image = _center_on_square_canvas(enhanced_rgb_image)
    # Se redimensiona a la resolución esperada por MobileNetV2.
    resized_image = centered_image.resize(
        (TRANSFER_IMAGE_WIDTH, TRANSFER_IMAGE_HEIGHT),
        resample=Image.Resampling.BICUBIC,
    )
    # Se convierte la imagen a array float32 para aplicar preprocess_input.
    image_array = np.array(resized_image, dtype=np.float32)
    # Se añade una dimensión de lote.
    image_batch = np.expand_dims(image_array, axis=0)
    # Se aplica la normalización oficial de MobileNetV2.
    processed_batch = preprocess_input(image_batch)
    # Se devuelve el lote listo para inferencia.
    return processed_batch