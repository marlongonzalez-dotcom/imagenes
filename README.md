# imagenes
proyecto 4 computacion 
readme_content = """
# Procesamiento de imágenes de torres de transmisión

**Autores:** Julio Bracamonte, Marlon González, Laura Sofía Alarcón.

## Objetivos del Proyecto
Este proyecto tiene como objetivos principales el procesamiento de imágenes capturadas por drones de torres de transmisión eléctrica y el entrenamiento de un modelo para detectar óxido en los soportes de los pararrayos.

## Librerías Utilizadas
Las librerías centrales para este proyecto incluyen:
- `zipfile`: Para la descompresión de datasets.
- `os`: Para la interacción con el sistema de archivos.
- `cv2` (OpenCV): Para lectura, preprocesamiento y manipulación de imágenes.
- `numpy`: Para operaciones numéricas, especialmente con arreglos.
- `matplotlib.pyplot`: Para la visualización de imágenes y resultados.
- `skimage.feature` (HOG, local_binary_pattern): Para la extracción de características de imágenes.
- `sklearn.svm` (SVC): Para la implementación del clasificador de Máquinas de Vectores de Soporte.
- `sklearn.model_selection` (StratifiedKFold): Para la división de datos en validación cruzada estratificada.
- `sklearn.preprocessing` (StandardScaler): Para escalar las características antes del entrenamiento del modelo.
- `sklearn.pipeline` (make_pipeline): Para construir pipelines de procesamiento y modelado.
- `sklearn.metrics` (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix): Para la evaluación del rendimiento del modelo.

## Detalles del Dataset
Los datos provienen de datasets de torres de transmisión y detección de fallas, subidos como archivos `.zip` a Google Colab debido a las restricciones de carpetas. La librería `zipfile` se utiliza para extraer estos archivos temporalmente.

**Enlaces a los datasets originales:**
- Torres de Transmisión: [https://doi.org/10.48550/arXiv.2311.02747](https://doi.org/10.48550/arXiv.2311.02747)
- Detección de fallas: [https://doi.org/10.1080/01431161.2023.2283900](https://doi.org/10.1080/01431161.2023.2283900)

El dataset se organiza en dos categorías principales:
- **`rutas_torres`**: 25 imágenes de torres de transmisión.
- **`rutas_soporte_pararrayos`**: 910 imágenes de soportes de pararrayos, utilizadas para la detección de óxido. Estas se dividen en subcarpetas `good` y `rust`.

## Preprocesamiento de Imágenes
Se definen dos funciones clave para el procesamiento de imágenes:

### `lectura_y_tam(path, size=(512,512))`
Lee una imagen desde la ruta especificada y la redimensiona a un tamaño estándar (por defecto, 512x512 píxeles) utilizando interpolación `cv.INTER_AREA`. Esto asegura que todas las imágenes tengan las mismas dimensiones para un procesamiento consistente.

### `preprocesar(img, blur_kernel=(5,5), canny_thresh=(30,125))`
Aplica una serie de transformaciones a la imagen:
1.  **Escala de grises**: Convierte la imagen de BGR a escala de grises usando `cv.cvtColor(img, cv.COLOR_BGR2GRAY)`.
2.  **Suavizado Gaussiano**: Aplica un filtro Gaussiano (`cv.GaussianBlur`) para reducir el ruido. Se utiliza un `blur_kernel` de tamaño impar (por defecto, 5x5) para asegurar que el píxel central sea considerado en el suavizado.
3.  **Detección de bordes Canny**: Utiliza el algoritmo Canny (`cv.Canny`) para resaltar los contornos, con umbrales `canny_thresh` (por defecto, 30 y 125) para determinar los bordes.

**Visualización de ejemplo:**
(Aquí se mostraría una imagen con las cuatro etapas: Original, Escala de grises, Suavizado, Contorno)

## Extracción de Rasgos (Features)
Para la detección de óxido, se utilizan dos métodos principales de extracción de rasgos, concatenados para formar un vector de características robusto:

### Histogram of Oriented Gradients (HOG)
La función `hog` de `skimage.feature` calcula la dirección y magnitud del gradiente de cada píxel, dividiendo la imagen en celdas y bloques. Parámetros clave:
-   `orientations=9`
-   `pixels_per_cell=(16, 16)`
-   `cells_per_block=(2, 2)`
-   `block_norm='L2-Hys'`
Este método es efectivo para capturar la forma de los objetos en la imagen.

### Local Binary Patterns (LBP)
La función `local_binary_pattern` de `skimage.feature` compara cada píxel con sus vecinos para generar un patrón binario, que luego se compila en un histograma. Parámetros clave:
-   `P=8` (número de puntos vecinos)
-   `R=1` (radio del círculo)
-   `method='uniform'`
LBP es útil para describir la textura de la imagen, diferenciando entre superficies lisas y rugosas (como el óxido).

Ambos vectores de características (HOG y LBP) se concatenan (`np.hstack`) para crear un vector de rasgos combinado que describe tanto la forma como la textura de las piezas.

## Entrenamiento y Evaluación del Modelo

El objetivo es clasificar los soportes de pararrayos como 'BUEN ESTADO' (0) o 'OXIDADO' (1) utilizando un modelo de Machine Learning. El proceso sigue los siguientes pasos:

### Preparación de Datos
-   Las imágenes de soportes de pararrayos se dividen en dos clases: `good` y `rust`.
-   Se extraen los rasgos HOG y LBP de cada imagen usando `extraer_rasgos`.
-   Los rasgos (`X`) y las etiquetas (`y`) se almacenan como arreglos de NumPy.

### Metodología de Entrenamiento y Evaluación
Se emplea la **Validación Cruzada Estratificada K-Fold** con `k=5` (`StratifiedKFold`). Esta técnica divide el dataset en 5 'folds', asegurando que la proporción de clases (buen estado vs. oxidado) se mantenga en cada división. El modelo se entrena en 4 folds y se evalúa en el quinto, rotando el fold de evaluación hasta que todos hayan sido utilizados.

El modelo utilizado es una **Máquina de Vectores de Soporte (SVC)** con un `kernel="linear"`, integrado en un `pipeline` junto con un `StandardScaler` para normalizar los datos. `make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))`.

### Métricas de Evaluación
Para cada fold, se calculan las siguientes métricas:
-   **Accuracy**: Proporción de predicciones correctas sobre el total. Puede ser engañosa en datasets desbalanceados.
-   **Precision**: De todas las predicciones positivas (oxidado), cuántas fueron realmente correctas. `TP / (TP + FP)`.
-   **Recall**: De todos los casos positivos reales (oxidado), cuántos fueron detectados correctamente por el modelo. `TP / (TP + FN)`.
-   **F1-score**: Media armónica de Precision y Recall. Es una métrica más robusta para datasets desbalanceados.

**Conceptos clave:**
-   **Matriz de Confusión**: Herramienta para visualizar el rendimiento del modelo, categorizando las predicciones en True Positive (TP), False Positive (FP), True Negative (TN) y False Negative (FN).
    -   `True Positive (TP)`: El modelo predice 'oxidado' y es correcto.
    -   `False Positive (FP)`: El modelo predice 'oxidado' pero en realidad es 'buen estado'.
    -   `True Negative (TN)`: El modelo predice 'buen estado' y es correcto.
    -   `False Negative (FN)`: El modelo predice 'buen estado' pero en realidad está 'oxidado'.

## Resultados Finales (Métricas Promedio)
Después de completar los 5 folds de validación cruzada, se obtienen las siguientes métricas promedio:
-   Accuracy promedio: 0.9659
-   Precision promedio: 0.9490
-   Recall promedio: 0.9576
-   F1-score promedio: 0.9531

Estos resultados demuestran un alto rendimiento del modelo en la detección de óxido en los soportes de pararrayos.

## Clasificación Interactiva por Índice

La función `clasificar_por_indice(idx)` permite al usuario probar el modelo con una imagen específica del dataset total de soportes de pararrayos, simplemente proporcionando su índice (número de 1 a 910). La función realiza lo siguiente:
1.  Toma un índice `idx` como entrada.
2.  Obtiene la ruta de la imagen correspondiente del `rutas_total`.
3.  Extrae los rasgos de la imagen usando `extraer_rasgos`.
4.  Utiliza el modelo entrenado para predecir si la pieza está 'OXIDADO' o en 'BUEN ESTADO'.
5.  Muestra la imagen original junto con la predicción del modelo.

Esto facilita la revisión manual y visual de las predicciones del modelo.

## Conclusiones
El proyecto ha cumplido exitosamente sus dos objetivos principales: demostrar el procesamiento de imágenes para torres de transmisión y entrenar un modelo robusto para la detección de óxido en soportes de pararrayos. La librería OpenCV fue fundamental en todas las etapas, desde el preprocesamiento básico hasta la preparación de datos para el entrenamiento. Las métricas de evaluación proporcionaron una sólida confirmación del buen desempeño del modelo. La inclusión de la función `clasificar_por_indice` mejora la usabilidad al permitir una verificación interactiva y sencilla de las predicciones.
"""
