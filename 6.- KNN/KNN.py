import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# 1. Cargar el archivo CSV (asegúrate de que el archivo esté en el mismo directorio o proporciona la ruta correcta)
df = pd.read_csv("archivo.csv", delimiter=";")

# 2. Preprocesamiento
# Definir las características (features) y la variable objetivo (target)
X = df[['ELECTORES', 'VOTOS NO', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS IMPUGNADOS', 'VOTOS TOTAL']]  # Características
y = df['VOTOS SI'] > df['VOTOS NO']  # 1 si "VOTOS SI" es mayor que "VOTOS NO", 0 si no

# 3. Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Especificar la carpeta de salida '6.- KNN'
output_folder = '6.- KNN'

# 5. Comprobar si la carpeta '6.- KNN' existe, si no existe la creamos
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  # Crear la carpeta si no existe
    print(f"La carpeta '{output_folder}' no existía, pero la hemos creado.")
else:
    print(f"La carpeta '{output_folder}' ya existe, procederemos con la guardada de los resultados.")

# 6. Evaluar la precisión del modelo para diferentes valores de K
k_values = range(1, 21)  # Probar para valores de K entre 1 y 20
accuracies = []

for k in k_values:
    # Crear el clasificador KNN con el valor de K actual
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Predecir los valores en el conjunto de prueba
    y_pred = knn.predict(X_test)
    
    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# 7. Graficar la precisión en función de los valores de K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', color='b', linestyle='-', markersize=8)
plt.title('Precisión del Modelo KNN para Diferentes Valores de K')
plt.xlabel('Número de Vecinos (K)')
plt.ylabel('Precisión')
plt.xticks(k_values)
plt.grid(True)

# 8. Guardar la imagen del gráfico de precisión en la carpeta
output_image_path = os.path.join(output_folder, 'KNN_precision_vs_K.png')
plt.savefig(output_image_path)  # Guardar la imagen en la carpeta especificada
plt.close()  # Cerrar la figura

# Confirmación de que la imagen se guardó correctamente
print(f"El gráfico de precisión del modelo KNN se ha guardado en: {output_image_path}")

# 9. Obtener el mejor valor de K (el que da la mayor precisión)
best_k = k_values[accuracies.index(max(accuracies))]
print(f"El mejor valor de K es: {best_k}")

# 10. Crear y mostrar la matriz de confusión para el mejor valor de K
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred_best_k = best_knn.predict(X_test)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred_best_k)

# Mostrar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues')
plt.title(f'Matriz de Confusión para K={best_k}')

# 11. Guardar la matriz de confusión como imagen
cm_image_path = os.path.join(output_folder, f'ConfusionMatrix_K{best_k}.png')
plt.savefig(cm_image_path)  # Guardar la imagen de la matriz de confusión
plt.close()

# Confirmación de que la imagen de la matriz de confusión se guardó correctamente
print(f"La matriz de confusión para K={best_k} se ha guardado en: {cm_image_path}")
