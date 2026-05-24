# Resumen ejecutivo para exposición

## Título del proyecto

Sistema inteligente de clasificación automática de prendas con CNN y Fashion MNIST.

## Problema abordado

Se ha desarrollado una solución capaz de identificar automáticamente diez tipos de prendas de vestir a partir de imágenes en escala de grises de `28x28` píxeles.

## Solución implementada

- Carga del dataset Fashion MNIST.
- Preprocesamiento mediante normalización y adaptación a formato `28x28x1`.
- Entrenamiento de una CNN con dos capas convolucionales.
- Evaluación con `accuracy`, `loss` y matriz de confusión.
- Generación de figuras académicas.
- Predicción sobre imagen externa.
- Interfaz gráfica con Tkinter.

## Arquitectura de la CNN

`Conv2D(32,3x3,relu) → MaxPool(2x2) → Conv2D(64,3x3,relu) → MaxPool(2x2) → Flatten → Dense(128,relu) → Dropout(0.3) → Dense(10,softmax)`

## Hiperparámetros principales

- Optimizador: Adam
- Función de pérdida: sparse categorical crossentropy
- Batch size: 32
- Épocas: 10

## Resultados esperados a mostrar en la defensa

- Curva de accuracy de entrenamiento y validación.
- Curva de pérdida de entrenamiento y validación.
- Accuracy final sobre test.
- Matriz de confusión en CSV y en figura PNG.
- Ejemplos correctamente clasificados.
- Demostración con una imagen de muestra.

## Resultados obtenidos en la ejecución validada

- Accuracy final sobre test: **0.9106**
- Loss final sobre test: **0.2522**
- La clase `Pantalón` alcanzó una precisión especialmente alta.
- Las principales confusiones aparecieron entre `Camiseta o top`, `Jersey`, `Abrigo` y `Camisa`.
- La imagen de ejemplo generada desde el dataset se predijo correctamente tras aplicar el preprocesamiento final.

## Valor académico del trabajo

- Aplica CRISP-DM de forma explícita.
- Conecta teoría matemática y práctica de ingeniería.
- Presenta una arquitectura modular y mantenible.
- Ofrece un despliegue demostrativo con interfaz gráfica.

## Recomendación para la exposición oral

Conviene comenzar explicando el problema, continuar con el flujo CRISP-DM, mostrar la arquitectura de la CNN, presentar las métricas y terminar con la demostración práctica en la GUI.