# Importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Cargar el archivo CSV desde una ubicación local (reemplaza 'archivo.csv' con tu ruta)
data = pd.read_csv('archivo.csv', delimiter=';')

# Convertir columnas categóricas a numéricas si es necesario
labelencoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = labelencoder.fit_transform(data[column])

# Suponiendo que las dos primeras columnas son las características y la tercera es la etiqueta
X = data[['VOTOS SI', 'VOTOS NO']].values  # Usar las columnas 'VOTOS SI' y 'VOTOS NO' para visualización
y = data.iloc[:, -1].values   # Seleccionar la última columna como etiquetas

# Escalar las características para mejorar el rendimiento del SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definición de kernels y valores de C a analizar
kernels = ['linear', 'rbf', 'poly']
C_values = [0.1, 1, 10, 100]

# Crear subplots para visualizar todas las combinaciones
fig, axes = plt.subplots(len(kernels), len(C_values), figsize=(20, 12))

# Lista para almacenar los resultados
resultados = []

for i, kernel in enumerate(kernels):
    for j, C in enumerate(C_values):
        # Entrenar el modelo SVM con el kernel y C actuales
        if kernel == 'poly':
            model = SVC(kernel=kernel, C=C, degree=3)
        else:
            model = SVC(kernel=kernel, C=C)
        model.fit(X_train, y_train)

        # Crear malla para graficar los límites de decisión
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

        # Predecir en la malla
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Graficar límites de decisión
        ax = axes[i, j]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm, alpha=0.8)
        ax.set_title(f"Kernel={kernel}, C={C}")
        ax.set_xlabel("VOTOS SI")
        ax.set_ylabel("VOTOS NO")
        ax.legend(*scatter.legend_elements(), title="Clases")

        # Guardar el resultado en la lista
        precision = model.score(X_test, y_test)
        resultados.append(f"| SVM Modelo con {kernel} | {kernel} | {C} | {precision:.2f} |")

# Ajustar espacio entre subplots
plt.tight_layout()

# Guardar el gráfico en la carpeta 9.- MaquinaSoporteVectorial
output_dir = '9.- MaquinaSoporteVectorial'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'RESULTADO_DEL_EJERCICIO_2.png')
plt.savefig(output_path)

# Mostrar el gráfico
plt.show()

# Escribir los resultados en el archivo README.md
readme_path = os.path.join(output_dir, 'README.md')
with open(readme_path, 'a') as f:
    f.write("\n# Resultados detallados del modelo SVM\n")
    f.write("| **Modelo** | **Kernel** | **Valor de C** | **Precisión** |\n")
    f.write("|------------|------------|----------------|---------------|\n")
    for resultado in resultados:
        f.write(resultado + "\n")
    f.write("\n![Resultado del Ejercicio 2](RESULTADO_DEL_EJERCICIO_2.png)\n")

print(f"Resultados guardados en {readme_path}")