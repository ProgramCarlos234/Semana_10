from __future__ import annotations

# Se importa tkinter para tipar contenedores base.
import tkinter as tk
# Se importa ttk para construir widgets con estilo nativo.
from tkinter import ttk


# Esta función crea el título principal de una sección o ventana.
def create_title(parent: tk.Misc, text: str) -> ttk.Label:
    """Crear una etiqueta de título destacada para la interfaz gráfica."""

    # Se devuelve una etiqueta con tipografía grande y negrita.
    return ttk.Label(parent, text=text, font=("Segoe UI", 18, "bold"))


# Esta función crea una etiqueta de cuerpo de texto estándar.
def create_body_label(parent: tk.Misc, text: str, wraplength: int = 800) -> ttk.Label:
    """Crear una etiqueta informativa con estilo homogéneo en la GUI."""

    # Se devuelve una etiqueta con tipografía legible y ajuste automático.
    return ttk.Label(parent, text=text, font=("Segoe UI", 11), wraplength=wraplength, justify="left")


# Esta función crea un botón estándar reutilizable en toda la GUI.
def create_button(parent: tk.Misc, text: str, command) -> ttk.Button:
    """Crear un botón con el estilo base del proyecto."""

    # Se crea y devuelve el botón asociado al comando recibido.
    return ttk.Button(parent, text=text, command=command)


# Esta función crea un marco con relleno homogéneo.
def create_frame(parent: tk.Misc, padding: int | tuple[int, int, int, int] = 12) -> ttk.Frame:
    """Crear un contenedor ttk con padding uniforme para componer la GUI."""

    # Se devuelve el marco listo para ser colocado en la ventana.
    return ttk.Frame(parent, padding=padding)