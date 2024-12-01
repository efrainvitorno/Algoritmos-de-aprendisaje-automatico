import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# Cargar los datos desde un archivo CSV (delimitador ";")
df = pd.read_csv("archivo.csv", delimiter=";")

# Seleccionar las características relevantes y la columna objetivo
features = ['ELECTORES', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS IMPUGNADOS', 'VOTOS TOTAL']
target = 'VOTOS SI'

# Extraer las características y el objetivo
X = df[features].values
y = df[target].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Entrenar el modelo de regresión lineal
lr = LinearRegression()
lr.fit(X_train, y_train)

# Realizar predicciones
y_pred = lr.predict(X_test)

# Calcular métricas de desempeño
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Obtener los coeficientes (pesos) y el intercepto (término independiente)
coeficientes = lr.coef_
intercepto = lr.intercept_

# Crear la ecuación del modelo
ecuacion = f"### Ecuación del Modelo:\n\nVOTOS SI = {intercepto:.2f} + " + " + ".join(
    [f"{coef:.2f} * {feature}" for coef, feature in zip(coeficientes, features)]
)

# Crear las métricas como texto
metricas = f"""
### Métricas del Modelo:

- **MAE:** {mae:.2f}
- **MSE:** {mse:.2f}
- **R2 Score:** {r2:.2f}
"""

# Combinar los resultados en una variable
resultados_md = ecuacion + "\n\n" + metricas

# Guardar los resultados en un archivo .md
with open("RegresionLinealMultiple\README.md", "w", encoding="utf-8") as archivo:
    archivo.write(resultados_md)

# Mostrar mensaje de éxito
print("Resultados guardados en 'resultados_modelo.md'")

# Visualización de los datos en 3D (opcional)
fig = px.scatter_3d(df, x='ELECTORES', y='VOTOS BLANCOS', z='VOTOS SI', title="Visualización de votos")
fig.show()
