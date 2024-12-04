import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import joblib  # Para guardar el modelo

# 1. Cargar el archivo CSV
try:
    df = pd.read_csv("archivo.csv", delimiter=";")
except FileNotFoundError:
    print("El archivo no se encontró. Verifica la ruta.")
    exit()

# 2. Verificación y preprocesamiento
if df.isnull().sum().any():
    print("El dataset contiene valores nulos. Por favor, limpia los datos antes de proceder.")
    exit()

# Definir las características (features) y la variable objetivo (target)
X = df[['ELECTORES', 'VOTOS NO', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS IMPUGNADOS', 'VOTOS TOTAL']]
y = (df['VOTOS SI'] > df['VOTOS NO']).astype(int)  # Convertir a 0 y 1

# 3. Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Crear y entrenar el modelo de Árbol de Decisión con hiperparámetros
model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10)
model.fit(X_train, y_train)

# 5. Evaluación del modelo
y_pred = model.predict(X_test)
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))

# 6. Definir la carpeta "8.- Arbol de decision"
output_folder = "8.- Arbol de decision"
os.makedirs(output_folder, exist_ok=True)  # Asegurar que la carpeta existe, pero no crear otra

# 7. Guardar el modelo entrenado
model_path = os.path.join(output_folder, 'modelo_arbol_decision.pkl')
joblib.dump(model, model_path)
print(f"El modelo se ha guardado en: {model_path}")

# 8. Visualizar y guardar el árbol de decisión
output_image_path = os.path.join(output_folder, 'Arbol_de_Decision.png')
plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No Revocatoria', 'Revocatoria'], rounded=True)
plt.title("Árbol de Decisión")
plt.savefig(output_image_path)
plt.close()

print(f"El árbol de decisión se ha guardado en: {output_image_path}")
