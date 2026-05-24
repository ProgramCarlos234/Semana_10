from __future__ import annotations

# Se importa argparse para definir una interfaz de línea de comandos limpia.
import argparse
# Se importa json para mostrar resultados resumidos en consola.
import json

# Se importa la rutina de evaluación del modelo guardado.
from src.evaluator import evaluate_custom_model, evaluate_saved_model
# Se importa el lanzador de la interfaz gráfica.
from src.gui.app import launch_gui
# Se importa la rutina completa de entrenamiento.
from src.trainer import train_and_save_model, train_custom_model


# Esta función construye el parser de argumentos del programa principal.
def build_parser() -> argparse.ArgumentParser:
    """Crear el parser con los subcomandos train, evaluate y gui."""

    # Se crea el parser principal con una descripción académica del proyecto.
    parser = argparse.ArgumentParser(
        description="Sistema inteligente de clasificación de prendas con CNN y Fashion MNIST.",
    )
    # Se crean subparsers obligatorios para los distintos modos de ejecución.
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Se define el subcomando de entrenamiento.
    subparsers.add_parser("train", help="Entrenar la CNN y generar todos los artefactos.")
    # Se define el subcomando de evaluación.
    subparsers.add_parser("evaluate", help="Evaluar el modelo entrenado y guardar métricas.")
    # Se define el subcomando de entrenamiento con dataset propio.
    custom_train_parser = subparsers.add_parser(
        "train_custom",
        help="Entrenar un modelo de transferencia con un dataset propio estructurado por carpetas.",
    )
    # Se añade la ruta del dataset personalizado como argumento configurable.
    custom_train_parser.add_argument(
        "--data-dir",
        default="data/custom_dataset",
        help="Directorio raíz con estructura ImageFolder, por ejemplo data/custom_dataset/camisa/*.jpg",
    )
    # Se define el subcomando de evaluación del modelo de transferencia.
    custom_evaluate_parser = subparsers.add_parser(
        "evaluate_custom",
        help="Evaluar el modelo de transferencia entrenado con un dataset propio.",
    )
    # Se añade también la ruta del dataset, necesaria para reconstruir la validación.
    custom_evaluate_parser.add_argument(
        "--data-dir",
        default="data/custom_dataset",
        help="Directorio raíz con estructura ImageFolder usado para entrenar y validar.",
    )
    # Se define el subcomando de interfaz gráfica.
    subparsers.add_parser("gui", help="Lanzar la interfaz gráfica Tkinter.")
    # Se devuelve el parser ya configurado.
    return parser


# Esta función actúa como punto de entrada coordinando cada modo de ejecución.
def main() -> None:
    """Interpretar el subcomando solicitado y ejecutar la acción correspondiente."""

    # Se construye el parser principal.
    parser = build_parser()
    # Se leen los argumentos introducidos por el usuario.
    args = parser.parse_args()
    # Si el usuario pide entrenar, se ejecuta el flujo completo de entrenamiento.
    if args.command == "train":
        # Se entrena el modelo y se recopila un resumen de la ejecución.
        summary = train_and_save_model()
        # Se imprime un resumen JSON para facilitar la trazabilidad.
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    # Si el usuario pide evaluar, se ejecuta la evaluación sobre el modelo ya guardado.
    if args.command == "evaluate":
        # Se ejecuta la evaluación y se obtienen las métricas agregadas.
        metrics = evaluate_saved_model()
        # Se imprimen las métricas en formato JSON legible.
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    # Si el usuario pide entrenar con un dataset propio, se ejecuta el flujo de transferencia.
    if args.command == "train_custom":
        # Se entrena el modelo profesional sin alterar el modelo simple existente.
        summary = train_custom_model(args.data_dir)
        # Se imprime el resumen para facilitar la trazabilidad.
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    # Si el usuario pide evaluar el modelo de transferencia, se regeneran métricas y figuras.
    if args.command == "evaluate_custom":
        # Se evalúa el modelo profesional usando la misma estructura de dataset.
        metrics = evaluate_custom_model(data_dir=args.data_dir)
        # Se imprimen las métricas en formato JSON legible.
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    # Si el usuario pide la interfaz gráfica, se lanza Tkinter.
    launch_gui()


# Este bloque garantiza que el script sólo actúe como programa al ejecutarse directamente.
if __name__ == "__main__":
    # Se llama al punto de entrada central de la aplicación.
    main()
