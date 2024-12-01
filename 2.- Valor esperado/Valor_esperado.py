import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("C:/Users/PC/Downloads/Resultados por distrito de la Revocatoria_Distrital.csv", encoding='ISO-8859-1', sep=';')


# Preprocesamiento de los datos
X = df[['ELECTORES', 'VOTOS NO', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS IMPUGNADOS', 'VOTOS TOTAL']]
y = df['VOTOS SI'] > df['VOTOS NO']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo de Árbol de Decisión
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Obtener las probabilidades de predicción para el conjunto de prueba
probabilidades = model.predict_proba(X_test)

# El valor esperado es la suma ponderada de las clases con sus probabilidades
# La clase 0 es 'No Revocatoria' y la clase 1 es 'Revocatoria'
# La probabilidad de clase 1 es la que nos interesa para calcular el valor esperado
valor_esperado = np.mean(probabilidades[:, 1])  # Promedio de las probabilidades de la clase 1

print(f"El valor esperado para la clase 'Revocatoria' es: {valor_esperado}")