from __future__ import annotations

# Se importa Path para admitir rutas de distintos tipos.
from pathlib import Path

# Se importan utilidades de Pillow para abrir y adaptar imágenes.
from PIL import Image, ImageOps, ImageTk

# Se importa el tamaño de vista previa definido en configuración.
from config.settings import GUI_PREVIEW_SIZE


# Esta función abre una imagen y la adapta a un tamaño de vista previa.
def build_preview_image(image_path: str | Path, target_size: tuple[int, int] = GUI_PREVIEW_SIZE) -> ImageTk.PhotoImage:
    """Crear una vista previa compatible con Tkinter respetando la proporción."""

    # Se abre la imagen original desde disco.
    image = Image.open(image_path)
    # Se convierte la imagen a un formato compatible estándar para evitar errores visuales.
    rgb_image = image.convert("RGB")
    # Se ajusta la imagen al tamaño máximo manteniendo su proporción.
    preview_image = ImageOps.contain(rgb_image, target_size)
    # Se transforma la imagen PIL en un objeto PhotoImage compatible con Tkinter.
    return ImageTk.PhotoImage(preview_image)