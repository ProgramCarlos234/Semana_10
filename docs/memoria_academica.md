# Memoria académica: sistema inteligente de clasificación automática de prendas con CNN y Fashion MNIST

## 1. Introducción

La clasificación automática de imágenes constituye una de las aplicaciones más representativas de la inteligencia artificial contemporánea. En el ámbito del reconocimiento visual, la identificación de prendas de vestir presenta un interés académico notable porque combina problemas reales de percepción, representación de características y toma de decisiones automatizada. En este proyecto se desarrolla un sistema completo para clasificar prendas utilizando el dataset **Fashion MNIST** y una **red neuronal convolucional (CNN)** implementada en **TensorFlow/Keras**.

El trabajo no se limita al entrenamiento del modelo, sino que abarca el ciclo completo de un proyecto aplicado: análisis del problema, preparación de los datos, diseño de la arquitectura, entrenamiento real, evaluación cuantitativa, generación de visualizaciones, demostración con imagen propia y despliegue en una interfaz gráfica construida con **Tkinter**. Esta visión integral permite defender el proyecto como una solución universitaria sólida y coherente.

## 2. Objetivos

### 2.1 Objetivo general

Desarrollar un sistema académico funcional para la clasificación automática de prendas de vestir, basado en una red neuronal convolucional entrenada sobre Fashion MNIST, documentando explícitamente el proceso bajo la metodología **CRISP-DM**.

### 2.2 Objetivos específicos

- Cargar y comprender el dataset Fashion MNIST.
- Preprocesar las imágenes para adaptarlas a la arquitectura propuesta.
- Implementar una CNN con la estructura aprobada en el plan del proyecto.
- Entrenar el modelo con `Adam`, `sparse_categorical_crossentropy`, `batch_size=32` y `epochs=10`.
- Evaluar el rendimiento mediante `accuracy`, `loss`, informe de clasificación y matriz de confusión.
- Generar figuras académicas de entrenamiento y resultados.
- Probar la solución con una imagen de ejemplo equivalente a una imagen de usuario.
- Construir una interfaz gráfica sencilla y demostrativa con Tkinter.
- Redactar documentación técnica extensa y formal en español de España.

## 3. Justificación

La relevancia del proyecto se sostiene en tres planos. En primer lugar, desde la perspectiva formativa, permite aplicar conceptos de inteligencia artificial, aprendizaje automático, aprendizaje profundo y visión por computador en un caso práctico y medible. En segundo lugar, desde la perspectiva metodológica, posibilita trabajar un flujo completo de proyecto, no sólo una prueba aislada de laboratorio. En tercer lugar, desde la perspectiva tecnológica, demuestra cómo un modelo entrenado puede integrarse en una herramienta usable por un usuario final.

Fashion MNIST resulta especialmente adecuado para fines académicos porque mantiene la simplicidad estructural de MNIST, pero introduce un reto de clasificación más realista: distinguir prendas que comparten contornos similares y variaciones intra-clase apreciables. Esto obliga a emplear arquitecturas capaces de extraer patrones espaciales, como las CNN.

## 4. Marco teórico

### 4.1 Inteligencia artificial

La inteligencia artificial es la disciplina que diseña sistemas capaces de realizar tareas que, tradicionalmente, requieren capacidades asociadas a la inteligencia humana, como percibir, aprender, razonar o decidir. En este proyecto, la capacidad central es la **percepción visual automatizada**, concretamente el reconocimiento de prendas en imágenes.

### 4.2 Machine Learning

El aprendizaje automático o *Machine Learning* es un subcampo de la inteligencia artificial que permite construir modelos que aprenden patrones a partir de datos. En lugar de programar reglas manuales para distinguir pantalones, bolsos o abrigos, se entrena un modelo a partir de ejemplos etiquetados.

### 4.3 Deep Learning

El aprendizaje profundo o *Deep Learning* se basa en redes neuronales de múltiples capas capaces de aprender representaciones jerárquicas. Las primeras capas suelen detectar bordes o texturas básicas, mientras que capas posteriores combinan esas señales en patrones más complejos, como formas características de una prenda.

### 4.4 Redes neuronales convolucionales

Las **CNN** están especialmente diseñadas para datos con estructura espacial, como las imágenes. Su principal ventaja frente a redes densas convencionales reside en que explotan la proximidad entre píxeles y aprenden filtros locales reutilizables. Las capas convolucionales detectan patrones, las capas de *pooling* reducen dimensionalidad y las capas densas realizan la clasificación final.

### 4.5 Dataset Fashion MNIST

Fashion MNIST es un conjunto de datos compuesto por:

- 60 000 imágenes de entrenamiento;
- 10 000 imágenes de prueba;
- resolución de `28x28` píxeles;
- escala de grises;
- 10 clases de prendas.

Su uso está ampliamente extendido en educación e investigación introductoria porque ofrece un equilibrio adecuado entre sencillez de manejo y complejidad visual moderada.

## 5. Metodología CRISP-DM

La metodología **CRISP-DM** (*Cross Industry Standard Process for Data Mining*) estructura el desarrollo del proyecto en seis fases iterativas.

### 5.1 Comprensión del negocio

En esta fase se definió el problema: construir un clasificador automático de prendas que fuese funcional, evaluable y demostrable en un entorno universitario. También se fijó el alcance: entrenamiento real, métricas, figuras, GUI y memoria académica.

### 5.2 Comprensión de los datos

Se analizó la naturaleza del dataset Fashion MNIST: imágenes pequeñas, monocromáticas y etiquetadas. Las clases muestran similitudes visuales importantes, especialmente entre `camiseta o top`, `camisa`, `jersey` y `abrigo`, lo que justifica el uso de una arquitectura que extraiga rasgos espaciales de forma robusta.

### 5.3 Preparación de los datos

La preparación consistió en:

- normalizar píxeles a `[0, 1]`;
- añadir un canal para pasar de `(28, 28)` a `(28, 28, 1)`;
- adaptar imágenes externas a escala de grises;
- redimensionar a `28x28`;
- normalizar y encapsular la entrada como lote.

### 5.4 Modelado

Se construyó una CNN con dos bloques convolucionales y dos capas densas finales. La selección de esta arquitectura se ajusta al plan aprobado y busca un equilibrio entre sencillez, interpretabilidad y rendimiento.

### 5.5 Evaluación

La evaluación incluye:

- `loss` sobre test;
- `accuracy` sobre test;
- informe de clasificación;
- matriz de confusión tabular;
- matriz de confusión visual;
- ejemplos clasificados correctamente.

### 5.6 Despliegue

La fase de despliegue se materializa mediante una aplicación de escritorio con Tkinter. Aunque no se trata de un despliegue web o empresarial, sí representa una forma clara y práctica de exposición del modelo entrenado ante un profesor o tribunal.

## 6. Arquitectura del sistema

El sistema sigue una arquitectura modular en la que cada fichero tiene una responsabilidad bien definida:

- `config/settings.py`: centraliza rutas, hiperparámetros y nombres de clases.
- `src/data_loader.py`: descarga y estructura Fashion MNIST.
- `src/preprocessing.py`: normaliza y transforma imágenes.
- `src/model_builder.py`: construye la CNN.
- `src/trainer.py`: entrena, guarda artefactos y ejecuta la demostración.
- `src/evaluator.py`: calcula métricas y genera evaluaciones.
- `src/predictor.py`: realiza inferencia sobre imágenes individuales.
- `src/visualizer.py`: produce curvas y figuras.
- `src/gui/`: encapsula la interfaz de usuario.
- `main.py`: ofrece los subcomandos `train`, `evaluate` y `gui`.

### 6.1 Diagrama textual de la red

```text
Entrada (28x28x1)
   ↓
Conv2D(32, 3x3, ReLU)
   ↓
MaxPooling2D(2x2)
   ↓
Conv2D(64, 3x3, ReLU)
   ↓
MaxPooling2D(2x2)
   ↓
Flatten
   ↓
Dense(128, ReLU)
   ↓
Dropout(0.3)
   ↓
Dense(10, Softmax)
   ↓
Clase predicha
```

## 7. Explicación matemática de las operaciones principales

### 7.1 Convolución 2D

La operación de convolución aplicada sobre una imagen puede expresarse como:

\[
S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i+m, j+n)\cdot K(m,n)
\]

donde:

- \(I\) es la imagen de entrada;
- \(K\) es el kernel o filtro aprendible;
- \(S(i,j)\) es el mapa de activación generado.

Cada filtro aprende a responder a patrones locales como bordes, esquinas o texturas.

### 7.2 Función de activación ReLU

Tras la convolución, se aplica habitualmente la función:

\[
ReLU(x) = \max(0, x)
\]

Esta activación introduce no linealidad y evita parte del problema de saturación presente en funciones como sigmoide o tangente hiperbólica.

### 7.3 MaxPooling

La operación de *max pooling* selecciona el valor máximo dentro de una ventana local. Para una ventana \(2 \times 2\):

\[
P(i,j) = \max \{ S(2i,2j), S(2i+1,2j), S(2i,2j+1), S(2i+1,2j+1) \}
\]

Esto reduce la dimensionalidad espacial, disminuye el coste computacional y aporta cierta invariancia ante pequeñas traslaciones.

### 7.4 Flatten

La operación *flatten* transforma un tensor tridimensional en un vector unidimensional:

\[
f: \mathbb{R}^{h \times w \times c} \rightarrow \mathbb{R}^{h \cdot w \cdot c}
\]

Esta transformación permite conectar la salida convolucional con capas densas tradicionales.

### 7.5 Capa densa

Una capa densa aplica una transformación afín seguida de una activación:

\[
y = \phi(Wx + b)
\]

donde:

- \(x\) es el vector de entrada;
- \(W\) es la matriz de pesos;
- \(b\) es el sesgo;
- \(\phi\) es la activación.

### 7.6 Softmax

La capa final usa *softmax* para obtener probabilidades sobre las diez clases:

\[
softmax(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
\]

La suma de las probabilidades resultantes es igual a 1.

## 8. Preprocesamiento y división train/test

El dataset ya proporciona una división oficial:

- entrenamiento: 60 000 imágenes;
- prueba: 10 000 imágenes.

Sobre esa base, se realiza un preprocesamiento mínimo pero imprescindible:

1. conversión a `float32`;
2. normalización de píxeles dividiendo entre 255;
3. adición del canal para obtener tensores `28x28x1`.

Para imágenes externas se añade además:

1. conversión a escala de grises;
2. inversión de intensidades para aproximar el estilo visual de Fashion MNIST;
3. redimensionado a `28x28`;
4. normalización;
5. encapsulado en un lote de tamaño 1.

## 9. Resultados experimentales

En la ejecución validada de este proyecto se obtuvieron los siguientes resultados principales sobre el conjunto de prueba:

- `accuracy = 0.9106`
- `loss = 0.2522`

Además, el sistema genera automáticamente:

- curva de `accuracy`;
- curva de `loss`;
- `evaluation_metrics.json`;
- `confusion_matrix.csv`;
- figura visual de la matriz de confusión;
- figura con ejemplos clasificados.

La interpretación de los resultados muestra una `accuracy` elevada para una arquitectura compacta y coherente con el alcance docente del proyecto. Los errores se concentran, como era esperable, entre clases visualmente similares, especialmente `camiseta o top`, `camisa`, `jersey` y `abrigo`.

La matriz de confusión permite comprobar no sólo cuántos errores existen, sino también **qué tipo de confusiones** comete el modelo. Esto es relevante académicamente porque aporta una lectura cualitativa del comportamiento del sistema.

## 10. Prueba con imagen propia

Como parte del entregable se genera una imagen de ejemplo en `data/sample_user_images/` a partir del propio conjunto de prueba. Esta imagen actúa como demostración controlada del flujo de inferencia. El resultado de su predicción queda registrado en `models/reports/sample_prediction.json`, donde aparecen:

- la ruta de la imagen utilizada;
- la clase real;
- la clase predicha;
- la probabilidad de la predicción principal;
- el top-3 de probabilidades.

En la ejecución final validada, la imagen de ejemplo fue clasificada correctamente, lo que confirma el funcionamiento extremo a extremo del flujo de inferencia: carga de imagen, preprocesamiento, predicción y registro de evidencias.

## 11. Conclusiones

El proyecto demuestra que una CNN relativamente compacta puede resolver con solvencia un problema de clasificación de prendas sobre Fashion MNIST. La solución desarrollada no sólo cumple con la dimensión algorítmica, sino también con la dimensión de ingeniería de software: estructura modular, reutilización de componentes, separación de responsabilidades, persistencia de artefactos y presentación visual de resultados.

Desde un punto de vista docente, la práctica resulta valiosa porque conecta teoría y aplicación. Conceptos como convolución, activación, pooling, optimización y generalización dejan de ser nociones abstractas y pasan a formar parte de un sistema tangible y demostrable.

## 12. Recomendaciones

- Entrenar en un entorno con CPU o GPU estable para reducir tiempos de ejecución.
- Mantener separadas las fases de entrenamiento, evaluación y demostración para mejorar la trazabilidad.
- Utilizar siempre las figuras y la matriz de confusión en la defensa oral, ya que aportan evidencia visual sólida.
- Explicar las limitaciones de generalización cuando se usan imágenes reales fuera del dominio del dataset.

## 13. Mejoras futuras

Como líneas de ampliación del proyecto se proponen:

- incorporar imágenes en color y conjuntos de datos más ricos;
- clasificar estilos adicionales como casual, deportivo o elegante;
- integrar **OpenCV** para segmentación y mejora del preprocesamiento;
- utilizar datasets más complejos como **DeepFashion**;
- aplicar aumento de datos (*data augmentation*);
- comparar la CNN base con arquitecturas más avanzadas;
- desplegar el sistema como aplicación web o servicio API.

## 14. Referencias

1. Han, X., Wu, Z., Wu, Z., Yu, R., & Davis, L. S. DeepFashion: Powering Robust Clothes Recognition and Retrieval.
2. Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*. MIT Press.
3. Chollet, F. *Deep Learning with Python*. Manning.
4. TensorFlow/Keras Documentation. https://www.tensorflow.org/
5. Fashion MNIST Dataset. https://github.com/zalandoresearch/fashion-mnist
6. Witten, I. H., Frank, E., Hall, M. A., & Pal, C. *Data Mining: Practical Machine Learning Tools and Techniques*.