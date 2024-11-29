<<<<<<< Updated upstream
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs, load_iris, load_wine, load_digits
import os
=======
# Importar las librerías necesarias
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
>>>>>>> Stashed changes

# Función para escribir resultados en el README.md
def escribir_resultados(texto):
    with open("MaquinaVectorial/README.md", "a", encoding="utf-8") as f:
        f.write(texto + "\n")

# Cargar el dataset archivo.csv
data = pd.read_csv("archivo.csv", delimiter=';')
X = data.drop(columns=['UBIGEO', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'AUTORIDAD EN CONSULTA'])
y = data['AUTORIDAD EN CONSULTA']

<<<<<<< Updated upstream
# Crear el archivo README.md (limpio) al iniciar
try:
    os.makedirs("MaquinaVectorial", exist_ok=True)
    with open("MaquinaVectorial/README.md", "w", encoding="utf-8") as f:
        f.write("# Proyecto Maquina Vectorial\n")
        f.write("Este proyecto explora la clasificación utilizando Máquinas de Vectores de Soporte (SVM) con diferentes kernels y parámetros.\n\n")
except Exception as e:
    print(f"Error al crear el archivo README.md: {e}")

# Ejercicio 1: Clasificación SVM con el dataset desde archivo CSV
guardar_en_readme("## Ejercicio 1: Clasificación SVM con el dataset desde archivo CSV")

# Cargar el dataset desde archivo CSV
data = pd.read_csv('archivo.csv', delimiter=';')

# Imprimir las primeras filas del DataFrame para verificar que se ha leído correctamente
print(data.head())

# Convertir columnas categóricas a numéricas usando one-hot encoding
data = pd.get_dummies(data, columns=['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'AUTORIDAD EN CONSULTA'])

# Asumiendo que las características están en todas las columnas menos la última
X = data.iloc[:, :-1].values
# Asumiendo que las etiquetas están en la última columna
y = data.iloc[:, -1].values

# Verificar las dimensiones de X e y
print(f"Dimensiones de X: {X.shape}")
print(f"Dimensiones de y: {y.shape}")

# Dividir en conjunto de entrenamiento y prueba
=======
# Dividir el conjunto de datos en entrenamiento y prueba
>>>>>>> Stashed changes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ejercicio 1: SVM con archivo.csv y kernel 'linear'
escribir_resultados("## Ejercicio 1: SVM con archivo.csv y kernel 'linear'")
model_linear = SVC(kernel='linear', random_state=42)
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)
precision_linear = accuracy_score(y_test, y_pred_linear)
escribir_resultados(f"Precisión con kernel 'linear': {precision_linear:.4f}")

# Ejercicio 2: SVM con archivo.csv y kernel 'rbf'
escribir_resultados("\n## Ejercicio 2: SVM con archivo.csv y kernel 'rbf'")
model_rbf = SVC(kernel='rbf', random_state=42)
model_rbf.fit(X_train, y_train)
y_pred_rbf = model_rbf.predict(X_test)
precision_rbf = accuracy_score(y_test, y_pred_rbf)
escribir_resultados(f"Precisión con kernel 'rbf': {precision_rbf:.4f}")

valores_C = [0.1, 1, 10, 100]
escribir_resultados("\nResultados con kernel 'rbf' y diferentes valores de C:")
for C in valores_C:
    model = SVC(kernel='rbf', C=C, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    escribir_resultados(f"Precisión con kernel 'rbf' y C={C}: {accuracy_score(y_test, y_pred):.4f}")

# Ejercicio 3: Visualización de límites de decisión con archivo.csv
escribir_resultados("\n## Ejercicio 3: Visualización de límites de decisión con archivo.csv")
# Seleccionar solo dos características para la visualización
X_train_vis = X_train.iloc[:, :2]
X_test_vis = X_test.iloc[:, :2]
model_visual = SVC(kernel='linear', C=1)
model_visual.fit(X_train_vis, y_train)
x_min, x_max = X_train_vis.iloc[:, 0].min() - 1, X_train_vis.iloc[:, 0].max() + 1
y_min, y_max = X_train_vis.iloc[:, 1].min() - 1, X_train_vis.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model_visual.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_train_vis.iloc[:, 0], X_train_vis.iloc[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title("Límites de decisión con SVM (Kernel 'linear')")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("MaquinaVectorial/limites_decision_linear.png")
plt.close()
escribir_resultados("![Límites de decisión con kernel 'linear'](limites_decision_linear.png)")

# Ejercicio 4: Optimización de hiperparámetros con archivo.csv
escribir_resultados("\n## Ejercicio 4: Optimización de hiperparámetros con archivo.csv")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)
escribir_resultados(f"Mejores parámetros: {grid.best_params_}")
escribir_resultados(f"Mejor precisión en validación cruzada: {grid.best_score_:.4f}")
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
escribir_resultados(f"Precisión en el conjunto de prueba con el mejor modelo: {accuracy_best:.4f}")

# Análisis y Comparación de Resultados
escribir_resultados("\n--- Análisis y Comparación de Resultados ---\n")
escribir_resultados("Resultados con Kernel 'linear':")
escribir_resultados(f"- Precisión: {precision_linear:.4f}\n")

escribir_resultados("Resultados con Kernel 'rbf':")
escribir_resultados(f"- Precisión: {precision_rbf:.4f}\n")

escribir_resultados("Resultados con Optimización de Hiperparámetros:")
escribir_resultados(f"- Mejores parámetros: {grid.best_params_}")
escribir_resultados(f"- Precisión en validación cruzada: {grid.best_score_:.4f}")
escribir_resultados(f"- Precisión en conjunto de prueba: {accuracy_best:.4f}\n")

<<<<<<< Updated upstream
# Experimentar con diferentes valores de C y kernel 'rbf' en datos sintéticos
valores_C = [0.1, 1, 10]
for C in valores_C:
    model_rbf_synthetic_c = SVC(kernel='rbf', C=C)
    model_rbf_synthetic_c.fit(X, y)
    Z_rbf_c = model_rbf_synthetic_c.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_rbf_c = Z_rbf_c.reshape(xx.shape)

    plt.contourf(xx, yy, Z_rbf_c, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(f"Límites de decisión con SVM (Kernel 'rbf', C={C})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    nombre_imagen = f"limites_rbf_C{C}.png"
    guardar_imagen(nombre_imagen)
    guardar_en_readme(f"### Límites de decisión con kernel 'rbf' y C={C}:")
    guardar_en_readme(f"![Límites de decisión con kernel 'rbf' y C={C}](limites_rbf_C{C}.png)")

guardar_en_readme("\n---")

# Ejercicio 3: Clasificación SVM con el dataset Iris completo y otros datasets
guardar_en_readme("## Ejercicio 3: Clasificación SVM con el dataset Iris completo y otros datasets")

# Paso 1: Cargar el dataset Iris completo
iris = load_iris()
X = iris.data
y = iris.target

# Paso 2: Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Paso 3: Entrenar un modelo SVM con kernel RBF
model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
model.fit(X_train, y_train)

# Paso 4: Hacer predicciones y calcular la precisión
y_pred = model.predict(X_test)
resultado_iris = accuracy_score(y_test, y_pred)
guardar_en_readme(f"Precisión multiclase con kernel 'rbf' en dataset Iris: {resultado_iris:.4f}")

# Experimentar con diferentes valores de gamma y observar los resultados
valores_gamma = [0.01, 0.1, 1, 10]
guardar_en_readme("\n### Resultados con diferentes valores de gamma en dataset Iris:")
for gamma in valores_gamma:
    model = SVC(kernel='rbf', C=1, gamma=gamma, random_state=42)
    model.fit(X_train, y_train)

    # Hacer predicciones y calcular la precisión
    y_pred = model.predict(X_test)
    resultado_gamma = accuracy_score(y_test, y_pred)
    guardar_en_readme(f"Precisión con kernel 'rbf' y gamma={gamma}: {resultado_gamma:.4f}")

# Probar con otros datasets: Wine y Digits
guardar_en_readme("\n### Resultados con otros datasets:")

# Dataset Wine
wine = load_wine()
X_wine = wine.data
y_wine = wine.target
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

model_wine = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
model_wine.fit(X_train_wine, y_train_wine)
y_pred_wine = model_wine.predict(X_test_wine)
resultado_wine = accuracy_score(y_test_wine, y_pred_wine)
guardar_en_readme(f"Precisión con dataset 'wine': {resultado_wine:.4f}")

# Dataset Digits
digits = load_digits()
X_digits = digits.data
y_digits = digits.target
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, test_size=0.3, random_state=42)

model_digits = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
model_digits.fit(X_train_digits, y_train_digits)
y_pred_digits = model_digits.predict(X_test_digits)
resultado_digits = accuracy_score(y_test_digits, y_pred_digits)
guardar_en_readme(f"Precisión con dataset 'digits': {resultado_digits:.4f}")

guardar_en_readme("\n---")

# Ejercicio 4: Optimización de Hiperparámetros con GridSearchCV en el dataset archivo.csv
guardar_en_readme("## Ejercicio 4: Optimización de Hiperparámetros con GridSearchCV en el dataset archivo.csv")

# Paso 1: Cargar el dataset desde archivo CSV
data = pd.read_csv('archivo.csv', delimiter=';')

# Convertir columnas categóricas a numéricas usando one-hot encoding
data = pd.get_dummies(data, columns=['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'AUTORIDAD EN CONSULTA'])

# Asumiendo que las características están en todas las columnas menos la última
X = data.iloc[:, :-1].values
# Asumiendo que las etiquetas están en la última columna
y = data.iloc[:, -1].values

# Paso 2: Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parte 1: Optimizar con kernel 'rbf'
param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Configurar y entrenar con GridSearchCV para el kernel 'rbf'
grid_rbf = GridSearchCV(SVC(), param_grid_rbf, refit=True, verbose=2, cv=5)
grid_rbf.fit(X_train, y_train)

# Mostrar los mejores parámetros y la precisión de la validación cruzada
mejores_parametros_rbf = grid_rbf.best_params_
mejor_precision_rbf = grid_rbf.best_score_
guardar_en_readme(f"\nMejores parámetros para el kernel 'rbf': {mejores_parametros_rbf}")
guardar_en_readme(f"Mejor precisión en validación cruzada con kernel 'rbf': {mejor_precision_rbf:.4f}")

# Evaluar el mejor modelo en el conjunto de prueba
best_model_rbf = grid_rbf.best_estimator_
y_pred_rbf = best_model_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
guardar_en_readme(f"Precisión en el conjunto de prueba con el mejor modelo (kernel='rbf'): {accuracy_rbf:.4f}")

# Parte 2: Cambiar el kernel a 'poly' o 'linear' y analizar los resultados
param_grid_kernels = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'poly']
}

# Configurar y entrenar con GridSearchCV para los kernels 'linear' y 'poly'
grid_kernels = GridSearchCV(SVC(), param_grid_kernels, refit=True, verbose=2, cv=5)
grid_kernels.fit(X_train, y_train)

# Mostrar los mejores parámetros y la precisión de la validación cruzada
mejores_parametros_kernels = grid_kernels.best_params_
mejor_precision_kernels = grid_kernels.best_score_
guardar_en_readme(f"\nMejores parámetros para los kernels 'linear' o 'poly': {mejores_parametros_kernels}")
guardar_en_readme(f"Mejor precisión en validación cruzada con kernels 'linear' o 'poly': {mejor_precision_kernels:.4f}")

# Evaluar el mejor modelo en el conjunto de prueba
best_model_kernels = grid_kernels.best_estimator_
y_pred_kernels = best_model_kernels.predict(X_test)
accuracy_kernels = accuracy_score(y_test, y_pred_kernels)
guardar_en_readme(f"Precisión en el conjunto de prueba con el mejor modelo (kernel='linear' o 'poly'): {accuracy_kernels:.4f}")

# Análisis y Comparación de Resultados
guardar_en_readme("\n--- Análisis y Comparación de Resultados ---\n")
guardar_en_readme("Resultados con Kernel 'rbf':")
guardar_en_readme(f"- Mejores parámetros: {mejores_parametros_rbf}")
guardar_en_readme(f"- Precisión en validación cruzada: {mejor_precision_rbf:.4f}")
guardar_en_readme(f"- Precisión en conjunto de prueba: {accuracy_rbf:.4f}\n")

guardar_en_readme("Resultados con Kernels 'linear' o 'poly':")
guardar_en_readme(f"- Mejores parámetros: {mejores_parametros_kernels}")
guardar_en_readme(f"- Precisión en validación cruzada: {mejor_precision_kernels:.4f}")
guardar_en_readme(f"- Precisión en conjunto de prueba: {accuracy_kernels:.4f}\n")

guardar_en_readme("Comentarios:")
guardar_en_readme("1. La precisión en validación cruzada muestra cómo de bien generaliza el modelo con diferentes configuraciones de hiperparámetros.")
guardar_en_readme("2. La precisión en el conjunto de prueba permite evaluar la capacidad del mejor modelo para predecir nuevas instancias.")
guardar_en_readme("3. Comparando los kernels 'rbf', 'linear', y 'poly', puedes observar cuál tiene mejor rendimiento en términos de generalización y precisión.")

guardar_en_readme("\n---")
guardar_en_readme("### Conclusión")
guardar_en_readme("Se observó que la precisión de los modelos SVM depende del kernel y del parámetro C. El modelo con kernel 'rbf' mostró un buen rendimiento en ambos ejercicios. Las gráficas generadas muestran claramente los límites de decisión para cada tipo de kernel.")

print("¡Ejercicio completado y README.md actualizado automáticamente!")
=======
# Comentarios para el Análisis
escribir_resultados("Comentarios:")
escribir_resultados("1. La precisión en validación cruzada muestra cómo de bien generaliza el modelo con diferentes configuraciones de hiperparámetros.")
escribir_resultados("2. La precisión en el conjunto de prueba permite evaluar la capacidad del mejor modelo para predecir nuevas instancias.")
escribir_resultados("3. Comparando los kernels 'linear' y 'rbf', puedes observar cuál tiene mejor rendimiento en términos de generalización y precisión.")
>>>>>>> Stashed changes
