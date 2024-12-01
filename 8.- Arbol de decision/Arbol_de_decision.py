import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os

# 1. Cargar el archivo CSV (asegúrate de que el archivo esté en el mismo directorio o proporciona la ruta correcta)
df = pd.read_csv("C:/Users/PC/Downloads/Resultados por distrito de la Revocatoria_Distrital.csv", encoding='ISO-8859-1', sep=';') 

# 2. Preprocesamiento
# Definir las características (features) y la variable objetivo (target)
X = df[['ELECTORES', 'VOTOS NO', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS IMPUGNADOS', 'VOTOS TOTAL']]  # Características
y = df['VOTOS SI'] > df['VOTOS NO']  # 1 si "VOTOS SI" es mayor que "VOTOS NO", 0 si no

# 3. Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Crear y entrenar el modelo de KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar n_neighbors según lo que necesites
knn.fit(X_train, y_train)

# 5. Crear la carpeta 'KNN' si no existe
output_folder = 'KNN'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 6. Guardar la imagen de los resultados en la carpeta
output_image_path = os.path.join(output_folder, 'KNN_resultados.png')

# 7. Graficar los resultados para visualización (opcional)
# Si deseas visualizar la distribución de los puntos en 2D (usando dos características, por ejemplo)
plt.figure(figsize=(10, 6))

# Aquí usamos solo dos características para la visualización
plt.scatter(X_train['ELECTORES'], X_train['VOTOS TOTAL'], c=y_train, cmap=plt.cm.Paired, edgecolors='k', s=30)
plt.title("Distribución de datos de entrenamiento para KNN")
plt.xlabel("ELECTORES")
plt.ylabel("VOTOS TOTAL")
plt.colorbar(label='Clase (Revocatoria)')

# Guardar la imagen del gráfico
plt.savefig(output_image_path)  # Guardar la imagen en la carpeta especificada
plt.close()  # Cerrar la figura

# Confirmación de que la imagen se guardó correctamente
print(f"Los resultados de KNN se han guardado en: {output_image_path}")
