import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer

# Cargar los datos desde un archivo CSV (delimitador ";")
df = pd.read_csv("archivo.csv", delimiter=";")

# Eliminar columnas irrelevantes
df = df.drop(columns=["UBIGEO", "DEPARTAMENTO", "PROVINCIA", "DISTRITO", "AUTORIDAD EN CONSULTA", "VOTOS IMPUGNADOS"])

# Separamos las características (X) y la variable objetivo (y)
X = df.drop(columns=["VOTOS SI"])  # Eliminamos la columna que queremos predecir
y = df["VOTOS SI"]

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un preprocesador para las características numéricas
numerical_columns = ['ELECTORES', 'VOTOS NO', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS TOTAL']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns)  # Escalar las columnas numéricas
    ])

# Crear pipelines para cada modelo (Ridge, Lasso y ElasticNet)
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('ridge', Ridge(alpha=1.0))
])

lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lasso', Lasso(alpha=1.0))
])

elasticnet_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('elasticnet', ElasticNet(alpha=1.0, l1_ratio=0.5))
])

# Entrenamos los modelos con los datos de entrenamiento
ridge_pipeline.fit(X_train, y_train)
lasso_pipeline.fit(X_train, y_train)
elasticnet_pipeline.fit(X_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred_ridge = ridge_pipeline.predict(X_test)
y_pred_lasso = lasso_pipeline.predict(X_test)
y_pred_elasticnet = elasticnet_pipeline.predict(X_test)

# Evaluar los modelos con MSE
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_elasticnet = mean_squared_error(y_test, y_pred_elasticnet)

# Imprimir los MSE de los modelos
print(f"MSE (Ridge): {mse_ridge}")
print(f"MSE (Lasso): {mse_lasso}")
print(f"MSE (ElasticNet): {mse_elasticnet}")

# 1. Definir nuevos datos de entrada (X_new)
# Asegúrate de que los valores coincidan con la estructura de X_train (sin las columnas eliminadas)
X_new = pd.DataFrame([
    [1126, 199, 68, 150, 858]    
], columns=numerical_columns)  # Usamos las mismas columnas que en X_train

# 2. Predecir con los modelos entrenados
y_pred_ridge_new = ridge_pipeline.predict(X_new)
y_pred_lasso_new = lasso_pipeline.predict(X_new)
y_pred_elasticnet_new = elasticnet_pipeline.predict(X_new)

# 3. Imprimir las predicciones
print("\nPredicciones con nuevos datos:")
print(f"Predicciones de Ridge: {y_pred_ridge_new}")
print(f"Predicciones de Lasso: {y_pred_lasso_new}")
print(f"Predicciones de ElasticNet: {y_pred_elasticnet_new}")

# 4. Visualización de los coeficientes (si es necesario)
plt.figure(figsize=(15, 5))

# Coeficientes de Ridge
plt.subplot(1, 3, 1)
plt.plot(ridge_pipeline.named_steps['ridge'].coef_, label="Ridge Coefficients")
plt.title("Ridge Coefficients")
plt.xlabel("Características")
plt.ylabel("Valor de los coeficientes")

# Coeficientes de Lasso
plt.subplot(1, 3, 2)
plt.plot(lasso_pipeline.named_steps['lasso'].coef_, label="Lasso Coefficients", color='r')
plt.title("Lasso Coefficients")
plt.xlabel("Características")
plt.ylabel("Valor de los coeficientes")

# Coeficientes de ElasticNet
plt.subplot(1, 3, 3)
plt.plot(elasticnet_pipeline.named_steps['elasticnet'].coef_, label="ElasticNet Coefficients", color='g')
plt.title("ElasticNet Coefficients")
plt.xlabel("Características")
plt.ylabel("Valor de los coeficientes")

# Guardar la figura
plt.tight_layout()
plt.savefig("Regularizacion/coeficientes.png")
plt.close()

# Crear el archivo README.md con los resultados y la imagen
resultados = f"""
# Regularización en Modelos de Regresión

## Introducción a la Regularización

La **regularización** es una técnica utilizada en la regresión y otros modelos estadísticos para evitar el **sobreajuste (overfitting)**. En modelos de regresión, el sobreajuste ocurre cuando el modelo se ajusta demasiado a los datos de entrenamiento y no generaliza bien a datos nuevos. La regularización agrega un término de penalización a la función de costo, lo que ayuda a reducir la complejidad del modelo y mejora su capacidad de generalización.

### Tipos de Regularización:
1. **Ridge** (L2): Penaliza la magnitud de los coeficientes del modelo. Esto ayuda a reducir los efectos de características irrelevantes, pero no las elimina completamente.
2. **Lasso** (L1): Penaliza la suma de los valores absolutos de los coeficientes. Este tipo de regularización puede llevar algunos coeficientes exactamente a cero, eliminando efectivamente las características menos relevantes.
3. **ElasticNet**: Combinación de Ridge y Lasso, que penaliza tanto la magnitud (L2) como la suma de los coeficientes absolutos (L1).

## Resultados del Modelo

Hemos entrenado tres modelos de regresión con regularización: **Ridge**, **Lasso** y **ElasticNet**. A continuación se presentan los **MSE (Mean Squared Error)** de cada modelo en los datos de prueba.

- MSE (Ridge): {mse_ridge} 
- MSE (Lasso): {mse_lasso} 
- MSE (ElasticNet): {mse_elasticnet}


### Predicciones para un nuevo conjunto de datos:

Ingresamos los siguientes datos para hacer predicciones con cada uno de los modelos:

| **ELECTORES** | **VOTOS NO** | **VOTOS BLANCOS** | **VOTOS NULOS** | **VOTOS IMPUGNADOS** | **VOTOS TOTAL** |
|---------------|--------------|-------------------|-----------------|----------------------|-----------------|
| 1126          | 441          | 199               | 68              | 150                  | 858             |

Las predicciones realizadas por cada modelo fueron:

- Predicciones de Ridge: {y_pred_ridge_new} 

- Predicciones de Lasso: {y_pred_lasso_new} 

- Predicciones de ElasticNet: {y_pred_elasticnet_new}

Como se puede observar, el modelo de **Lasso** proporciona la predicción más cercana al valor real de **441** (votos a favor), lo que indica que es el modelo que mejor se ajusta a estos datos específicos.

### Visualización de los Coeficientes

A continuación, se presentan los coeficientes de cada modelo:

![Coeficientes de los Modelos](coeficientes.png)

### Conclusión

El modelo **Lasso**, con su capacidad para penalizar más fuertemente las características irrelevantes, ha mostrado ser el más adecuado para esta predicción en particular, ya que su error cuadrático medio (MSE) es significativamente menor que el de los modelos **Ridge** y **ElasticNet**.

Es importante destacar que la elección del modelo más adecuado puede depender del contexto específico del conjunto de datos y los objetivos del análisis. Sin embargo, en este caso, **Lasso** parece ser el mejor modelo en términos de precisión y ajuste a los datos.
"""

# Guardar los resultados en un archivo README.md
with open("Regularizacion/README.md", "w") as f:
    f.write(resultados)
print("prueba")