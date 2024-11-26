# Importar librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset (ajustar la ruta del archivo según sea necesario)
file_path = 'archivo.csv'
data = pd.read_csv(file_path, sep=';')  # Ajustar el separador si es necesario

# Mostrar las primeras filas del dataset para explorarlo
print(data.head())

# Preprocesamiento: Seleccionar características (X) y etiquetas (y)
X = data.drop(columns=['VOTOS TOTAL'])  # Selección de todas las columnas excepto 'VOTOS TOTAL'
y = data['VOTOS TOTAL']  # 'VOTOS TOTAL' como etiqueta/clase

# Codificar etiquetas si son categóricas
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# Convertir características categóricas a numéricas
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = LabelEncoder().fit_transform(X[column])

# Escalar las características para SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo SVM con kernel 'rbf'
svm_model = SVC(kernel='rbf', random_state=42)

# Entrenar el modelo
svm_model.fit(X_train, y_train)

# Hacer predicciones
y_pred = svm_model.predict(X_test)

# Evaluar el modelo
classification_report_str = classification_report(y_test, y_pred)
accuracy_score_str = f"Exactitud del modelo con kernel 'rbf': {accuracy_score(y_test, y_pred)}"

print("Reporte de clasificación con kernel 'rbf':\n", classification_report_str)
print(accuracy_score_str)

# Escribir los resultados en el archivo README.md
with open('README.md', 'a') as f:
    f.write("\n## Resultados del modelo SVM\n")
    f.write("### Reporte de clasificación con kernel 'rbf':\n")
    f.write(f"{classification_report_str}\n")
    f.write(f"{accuracy_score_str}\n")