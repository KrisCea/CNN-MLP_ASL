# Clasificador de Lengua de Señas (ASL) — Arquitectura Híbrida CNN + MLP

Este proyecto implementa una red neuronal profunda de extremo a extremo en **Python (TensorFlow / Keras)** para la clasificación de imágenes del alfabeto en Lengua de Señas Estadounidense (*American Sign Language* - ASL) utilizando el dataset **Sign Language MNIST**.

El sistema acopla bloques convolucionales (**CNN**) para la extracción jerárquica de características espaciales junto con un Perceptrón Multicapa (**MLP**) regularizado para la inferencia multiclase, integrando un puente de reducción dimensional mediante `GlobalAveragePooling2D` para mitigar el sobreajuste.

---

## 📋 1. Características y Auditoría del Dataset

Los datos de origen se encuentran serializados en formato tabular (`sign_mnist_train.csv` y `sign_mnist_test.csv`)[cite: 6]:

* **Estructura de Entrada:** Cada registro consta de `785` columnas: `1` columna para la etiqueta (`label`) y `784` columnas (`pixel1`...`pixel784`) que representan una imagen en escala de grises de $28 \times 28$ píxeles en el rango `[0, 255]`[cite: 1, 6].
* **Volumen de Datos:**
  * **Entrenamiento Total:** `27,455` muestras[cite: 6] (divididas internamente de forma estratificada en `85%` entrenamiento y `15%` validación).
  * **Prueba Aislada (`test`):** `7,172` muestras[cite: 6].
* **Exclusión de Clases Dinámicas ($J$ y $Z$):** El alfabeto consta de 26 letras, pero las señas para **J** (índice `9`) y **Z** (índice `25`) requieren movimiento de la mano[cite: 2]. Al ser un dataset de capturas estáticas, el conjunto contiene exactamente **24 clases únicas**[cite: 1, 3, 6].
* **Corrección de Índices Discontinuos:** En los archivos CSV originales, las etiquetas van del `0` al `24` omitiendo el valor `9`[cite: 6]. El pipeline de preprocesamiento re-mapea automáticamente estas etiquetas a un índice continuo `[0..23]` antes de entrar a la función de pérdida `SparseCategoricalCrossentropy`, y revierte el mapeo a caracteres ASCII (`A-Y`, sin `J`) durante la inferencia.

---

## 🧠 2. Arquitectura del Modelo (Grafo Computacional)

El diseño evita el cuello de botella clásico de parámetros en la transición CNN $\rightarrow$ MLP:

1. **Aumento de Datos en GPU (*Data Augmentation*):**
   * Incluye `RandomRotation` ($\pm 8\%$), `RandomZoom` ($\pm 8\%$) y `RandomTranslation` ($\pm 5\%$).
   * **Restricción Semántica:** Se excluye deliberadamente el volteo horizontal (`RandomFlip`), dado que invertir la lateralidad de la mano altera la validez geométrica y direccional de múltiples señas (ej. letras *G* y *H*)[cite: 2, 3].
2. **Extractor Espacial (CNN):**
   * **Bloque 1:** `Conv2D(32, 3x3)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2x2)` (Salida: $14 \times 14 \times 32$).
   * **Bloque 2:** `Conv2D(64, 3x3)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2x2)` (Salida: $7 \times 7 \times 64$).
   * **Bloque 3 (Doble cascada):** Dos capas consecutivas `Conv2D(128, 3x3)` con `BatchNormalization` y `ReLU` sin *pooling* final (Salida: $7 \times 7 \times 128$), diseñadas para discriminar variaciones finas en la posición del pulgar (ej. pares críticos *M/N/S/T/A*)[cite: 2, 3].
3. **Puente Dimensional (`GlobalAveragePooling2D`):**
   * Sustituye la capa `Flatten()` tradicional (que generaría un vector de $6,272$ unidades y más de $1.6\text{M}$ de parámetros hacia el MLP) por un promedio espacial global de $128$ características ($33,024$ parámetros en la primera capa densa, logrando una reducción de **48.6x** en pesos en ese enlace).
4. **Clasificador (MLP):**
   * `Dense(256, ReLU)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dropout(0.4)`.
   * `Dense(128, ReLU)` $\rightarrow$ `Dropout(0.3)`.
   * **Capa de Salida:** `Dense(24, Softmax)`.

---

## 🛠️ 3. Requisitos e Instalación

Asegúrate de contar con **Python 3.10+** instalado. Se recomienda crear un entorno virtual limpio:

```bash
# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate        # En Linux/macOS
# venv\Scripts\activate         # En Windows

# Instalar dependencias
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow