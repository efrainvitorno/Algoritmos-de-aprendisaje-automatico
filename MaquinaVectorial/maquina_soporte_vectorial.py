import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs

# Función para guardar resultados en el README.md
def guardar_en_readme(texto):
    with open("MaquinaVectorial/README.md", "a") as f:
        f.write(texto + "\n")

# Función para guardar las gráficas
def guardar_imagen(nombre_imagen):
    plt.savefig(f"MaquinaVectorial/{nombre_imagen}")
    plt.close()

# Crear el archivo README.md (limpio) al iniciar
with open("MaquinaVectorial/README.md", "w") as f:
    f.write("# Proyecto Maquina Vectorial\n")
    f.write("Este proyecto explora la clasificación utilizando Máquinas de Vectores de Soporte (SVM) con diferentes kernels y parámetros.\n\n")

# Ejercicio 1: Clasificación SVM con el dataset Iris
guardar_en_readme("## Ejercicio 1: Clasificación SVM con el dataset Iris")

# Cargar el dataset Iris (usaremos solo dos clases para simplificar)
iris = datasets.load_iris()
X = iris.data[iris.target != 2]  # Seleccionar solo dos clases (0 y 1)
y = iris.target[iris.target != 2]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar un modelo SVM lineal
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones y calcular la precisión
y_pred = model.predict(X_test)
resultado_lineal = accuracy_score(y_test, y_pred)
guardar_en_readme(f"Precisión con kernel 'linear': {resultado_lineal:.4f}")

# Cambiar el kernel a 'rbf' y observar cómo afecta el rendimiento
model_rbf = SVC(kernel='rbf', random_state=42)
model_rbf.fit(X_train, y_train)
y_pred_rbf = model_rbf.predict(X_test)
resultado_rbf = accuracy_score(y_test, y_pred_rbf)
guardar_en_readme(f"Precisión con kernel 'rbf': {resultado_rbf:.4f}")

# Experimentar con diferentes valores de C
valores_C = [0.1, 1, 10, 100]
guardar_en_readme("### Resultados con kernel 'rbf' y diferentes valores de C:")
for C in valores_C:
    model_rbf_c = SVC(kernel='rbf', C=C, random_state=42)
    model_rbf_c.fit(X_train, y_train)
    y_pred_c = model_rbf_c.predict(X_test)
    resultado_c = accuracy_score(y_test, y_pred_c)
    guardar_en_readme(f"Precisión con kernel 'rbf' y C={C}: {resultado_c:.4f}")

guardar_en_readme("\n---")

# Ejercicio 2: Clasificación SVM con Datos Sintéticos
guardar_en_readme("## Ejercicio 2: Clasificación SVM con Datos Sintéticos")

# Generar datos sintéticos
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# Crear una malla para graficar los límites de decisión
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Modelo SVM con kernel 'linear'
model_linear = SVC(kernel='linear', C=1)
model_linear.fit(X, y)
Z_linear = model_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z_linear = Z_linear.reshape(xx.shape)

plt.contourf(xx, yy, Z_linear, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title("Límites de decisión con SVM (Kernel 'linear')")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
guardar_imagen("limites_lineal.png")
guardar_en_readme("### Límites de decisión con kernel 'linear':")
guardar_en_readme("![Límites de decisión con kernel 'linear'](MaquinaVectorial/limites_lineal.png)")

# Modelo SVM con kernel 'rbf'
model_rbf_synthetic = SVC(kernel='rbf', C=1)
model_rbf_synthetic.fit(X, y)
Z_rbf = model_rbf_synthetic.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = Z_rbf.reshape(xx.shape)

plt.contourf(xx, yy, Z_rbf, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title("Límites de decisión con SVM (Kernel 'rbf')")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
guardar_imagen("limites_rbf.png")
guardar_en_readme("### Límites de decisión con kernel 'rbf':")
guardar_en_readme("![Límites de decisión con kernel 'rbf'](MaquinaVectorial/limites_rbf.png)")

# Modelo SVM con kernel 'poly'
model_poly = SVC(kernel='poly', C=1, degree=3)
model_poly.fit(X, y)
Z_poly = model_poly.predict(np.c_[xx.ravel(), yy.ravel()])
Z_poly = Z_poly.reshape(xx.shape)

plt.contourf(xx, yy, Z_poly, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title("Límites de decisión con SVM (Kernel 'poly')")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
guardar_imagen("limites_poly.png")
guardar_en_readme("### Límites de decisión con kernel 'poly':")
guardar_en_readme("![Límites de decisión con kernel 'poly'](MaquinaVectorial/limites_poly.png)")

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
    guardar_en_readme(f"![Límites de decisión con kernel 'rbf' y C={C}](MaquinaVectorial/{nombre_imagen})")

guardar_en_readme("\n---")
guardar_en_readme("### Conclusión")
guardar_en_readme("Se observó que la precisión de los modelos SVM depende del kernel y del parámetro C. El modelo con kernel 'rbf' mostró un buen rendimiento en ambos ejercicios. Las gráficas generadas muestran claramente los límites de decisión para cada tipo de kernel.")

print("¡Ejercicio completado y README.md actualizado automáticamente!")
