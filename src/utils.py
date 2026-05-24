from __future__ import annotations

# Se importa json para guardar y cargar estructuras serializables.
import json
# Se importa random para fijar semillas de ejecución.
import random
# Se importa Path para manipular rutas del sistema de ficheros.
from pathlib import Path
# Se importa Any para tipar funciones genéricas.
from typing import Any

# Se importa NumPy para controlar la semilla numérica.
import numpy as np


# Esta función crea un directorio y todos sus padres cuando todavía no existen.
def ensure_directory(path: Path) -> None:
    """Crear un directorio de salida garantizando su existencia previa."""

    # Se crea la carpeta indicada junto con cualquier carpeta antecesora necesaria.
    path.mkdir(parents=True, exist_ok=True)


# Esta función crea todas las carpetas clave del proyecto para evitar errores posteriores.
def ensure_project_directories(paths: list[Path]) -> None:
    """Crear en bloque las carpetas necesarias para datos, modelos y documentación."""

    # Se recorre cada ruta recibida para asegurar su existencia una a una.
    for path in paths:
        # Se crea cada carpeta si todavía no existe.
        ensure_directory(path)


# Esta función guarda datos arbitrarios en formato JSON con UTF-8.
def save_json(path: Path, data: Any) -> None:
    """Persistir una estructura Python serializable en un archivo JSON legible."""

    # Se garantiza que la carpeta contenedora exista antes de escribir el archivo.
    ensure_directory(path.parent)
    # Se convierte la estructura en texto JSON con sangrado legible y sin escapar acentos.
    json_text = json.dumps(data, ensure_ascii=False, indent=2)
    # Se escribe el texto resultante en disco con codificación UTF-8.
    path.write_text(json_text, encoding="utf-8")


# Esta función carga un archivo JSON y devuelve su contenido en memoria.
def load_json(path: Path) -> Any:
    """Leer un archivo JSON desde disco y devolver la estructura de datos asociada."""

    # Se lee todo el contenido textual del archivo con codificación UTF-8.
    json_text = path.read_text(encoding="utf-8")
    # Se transforma el texto JSON en una estructura Python equivalente.
    return json.loads(json_text)


# Esta función fija semillas en módulos habituales para mejorar la repetibilidad.
def set_global_seed(seed: int) -> None:
    """Fijar la semilla global de Python y NumPy para estabilizar resultados."""

    # Se fija la semilla del generador aleatorio estándar de Python.
    random.seed(seed)
    # Se fija la semilla del generador aleatorio de NumPy.
    np.random.seed(seed)


# Esta función transforma una ruta en cadena absoluta para los informes finales.
def to_absolute_string(path: Path) -> str:
    """Obtener la ruta absoluta en formato cadena para reportes y trazabilidad."""

    # Se resuelve la ruta y se convierte a texto.
    return str(path.resolve())