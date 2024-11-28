import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# 4. Crear y entrenar el modelo de Árbol de Decisión
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Crear la carpeta 'Arbol de decision1' si no existe
output_folder = 'Arbol de decision1'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 6. Guardar la imagen del árbol de decisión en la carpeta
output_image_path = os.path.join(output_folder, 'Arbol_de_Decision.png')

# 7. Visualizar y guardar el árbol de decisión usando `plot_tree` de sklearn
plt.figure(figsize=(15, 10))  # Tamaño de la figura
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No Revocatoria', 'Revocatoria'], rounded=True)
plt.title("Árbol de Decisión")
plt.savefig(output_image_path)  # Guardar la imagen en la carpeta especificada
plt.close()  # Cerrar la figura

print(f"El árbol de decisión se ha guardado en: {output_image_path}")
