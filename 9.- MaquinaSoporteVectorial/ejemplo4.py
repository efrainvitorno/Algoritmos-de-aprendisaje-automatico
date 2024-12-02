import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Cargar el dataset con el delimitador ';'
data = pd.read_csv('archivo.csv', delimiter=';')

# Mostrar las primeras filas del dataframe
print(data.head())

# Convertir las columnas categóricas a variables dummy
data = pd.get_dummies(data, columns=['UBIGEO', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'AUTORIDAD EN CONSULTA'])

# Definir las características (X) y las columnas objetivo (y)
X = data.drop(['VOTOS NO', 'VOTOS SI', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS IMPUGNADOS', 'VOTOS TOTAL'], axis=1)
y = data[['VOTOS NO', 'VOTOS SI', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS IMPUGNADOS', 'VOTOS TOTAL']]

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Filtrar clases con solo 1 ejemplo
class_counts = y_train['VOTOS NO'].value_counts()
classes_to_remove = class_counts[class_counts == 1].index

# Filtrar el conjunto de entrenamiento para eliminar clases con solo 1 ejemplo
y_train_filtered = y_train[~y_train['VOTOS NO'].isin(classes_to_remove)]
X_train_filtered = X_train[~y_train['VOTOS NO'].isin(classes_to_remove)]

# Balancear las clases usando SMOTE con k_neighbors=1
smote = SMOTE(random_state=42, k_neighbors=1)  # Reducido a k_neighbors=1
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_filtered, y_train_filtered['VOTOS NO'])

# Definir el rango de hiperparámetros para GridSearchCV con menos combinaciones
param_grid = {
    'C': [1, 10],  # Reducido el rango de C
    'gamma': [1, 0.1],  # Reducido el rango de gamma
    'kernel': ['linear', 'poly']  # Mantengo solo dos kernels
}

# Configurar la búsqueda en cuadrícula con validación cruzada
cv = StratifiedKFold(n_splits=3)  # Reducido el número de splits para pruebas rápidas
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=cv)

# Entrenar el modelo y ajustar los hiperparámetros
grid.fit(X_train_balanced, y_train_balanced)

# Imprimir los mejores parámetros encontrados y la mejor precisión
print("Mejores parámetros encontrados:", grid.best_params_)
print("Mejor precisión en validación cruzada:", grid.best_score_)

# Obtener el resultado en formato tabla
resultado = pd.DataFrame(grid.cv_results_)

# Mostrar solo las columnas relevantes del resultado
print("\nResultados de GridSearchCV detallados:")
print(resultado[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

# Guardar el resumen de los resultados en un archivo .csv
resultado.to_csv('resultado_gridsearch.csv', index=False)

# Mostrar el dataframe con resultados en formato tabla (similar al ejemplo)
resultado_resumido = resultado[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score')

# Mostrar el resultado de forma ordenada
print("\nResultado ordenado por el mejor puntaje:")
print(resultado_resumido)

# Guardar el resumen de la tabla de resultados en formato Markdown para el README.md
resultado_markdown = resultado_resumido.to_markdown(index=False)

# Escribir el resultado en el README.md
with open('9.- MaquinaSoporteVectorial/README.md', 'a') as f:
    f.write("\n## Resultados de GridSearchCV\n")
    f.write("### Mejor combinación de hiperparámetros:\n")
    f.write(f"Mejores parámetros encontrados: {grid.best_params_}\n")
    f.write(f"Mejor precisión en validación cruzada: {grid.best_score_}\n")
    f.write("\n### Resultados de la búsqueda en cuadrícula:\n")
    f.write(resultado_markdown)
    f.write("\n")
