# Paso 1: Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Paso 2: Cargar el archivo CSV desde tu ruta local
df = pd.read_csv("C:/Users/PC/Downloads/Resultados por distrito de la Revocatoria_Distrital.csv", encoding='ISO-8859-1', sep=';')

# Ver las primeras filas del DataFrame para asegurarse que se ha cargado correctamente
print(df.head())

# Paso 3: Preprocesamiento de los datos
# Crear una columna 'REVOCATORIA' que será el objetivo: si 'VOTOS SI' > 'VOTOS NO' se considera 1, de lo contrario 0
df['REVOCATORIA'] = (df['VOTOS SI'] > df['VOTOS NO']).astype(int)

# Seleccionar las columnas de características (puedes ajustar según lo que necesites)
X = df[['VOTOS SI', 'VOTOS NO', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS IMPUGNADOS']]  # Variables predictoras
y = df['REVOCATORIA']  # Variable objetivo

# Paso 4: Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Paso 5: Normalizar las características (esto es importante para el algoritmo KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Paso 6: Aplicar el algoritmo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Paso 7: Hacer predicciones
y_pred = knn.predict(X_test_scaled)

# Paso 8: Evaluar el rendimiento del modelo
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Paso 9: Guardar la matriz de confusión como imagen
plt.figure(figsize=(6, 6))
plt.imshow(confusion_matrix(y_test, y_pred), cmap='Blues', interpolation='nearest')
plt.title('Matriz de Confusión')
plt.colorbar()
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.xticks([0, 1], ['No Revocatoria', 'Revocatoria'])
plt.yticks([0, 1], ['No Revocatoria', 'Revocatoria'])
plt.savefig('matriz_confusion.png')  # Guardar como imagen

# Paso 10: Guardar el gráfico de precisión como imagen
k_values = list(range(1, 21))
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    accuracy_scores.append(knn.score(X_test_scaled, y_test))

# Guardar el gráfico de precisión
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('Precisión del modelo KNN para diferentes valores de K')
plt.xlabel('Número de vecinos (K)')
plt.ylabel('Precisión')
plt.grid(True)
plt.savefig('precision_grafico.png')  # Guardar como imagen
