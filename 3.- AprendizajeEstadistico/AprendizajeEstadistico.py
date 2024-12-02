import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Escalar los datos para normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo KNN
knn = KNeighborsRegressor(n_neighbors=3)  # Usamos K=3 vecinos
knn.fit(X_train, y_train)

# Realizar predicciones
y_pred = knn.predict(X_test)

# Calcular métricas de desempeño
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Predicciones para nuevos datos
nuevo_dato = np.array([[500, 15, 5, 0, 340]])  # Ejemplo de nuevos datos
nuevo_dato_escalado = scaler.transform(nuevo_dato)
prediccion_nuevo_dato = knn.predict(nuevo_dato_escalado)

# Guardar los resultados en una variable para escribirlos en un archivo
resultados = f"""
El aprendizaje estadístico es un conjunto de métodos y técnicas que permiten a las má-
quinas (computadoras) aprender patrones a partir de datos para hacer predicciones, tomar
decisiones o identificar relaciones subyacentes en la información. Se basa en principios es-
tadísticos y matemáticos, y busca construir modelos que puedan generalizarse bien a datos
nuevos, es decir, ser precisos no solo en el conjunto de datos de entrenamiento, sino también
cuando se aplican a datos desconocidos.
---
### Tecnica KNN para la prediccion.
### Resultados del modelo KNN:
- MAE: {mae:.2f}
- MSE: {mse:.2f}
- R2 Score: {r2:.2f}

### Predicciones:
- Prediccion para nuevo dato (ELECTORES=500, VOTOS BLANCOS=15, VOTOS NULOS=5, VOTOS IMPUGNADOS=0, VOTOS TOTAL=340): {prediccion_nuevo_dato[0]:.2f}
"""

# Guardar los resultados y las predicciones en un archivo .md
with open("3.- AprendizajeEstadistico\README.md", "w") as f:
    f.write(resultados)

# Imprimir los resultados
print(resultados)
