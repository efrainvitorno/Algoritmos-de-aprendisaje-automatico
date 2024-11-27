# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Cargar y preparar datos
file_path = 'archivo.csv'
data = pd.read_csv(file_path, sep=';')

# Preprocesamiento
X = data[['UBIGEO', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'AUTORIDAD EN CONSULTA', 'ELECTORES', 'VOTOS SI', 'VOTOS NO', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS IMPUGNADOS']]
y = data['VOTOS TOTAL']

# Codificar variables categóricas
for column in X.select_dtypes(include=['object']).columns:
    X.loc[:, column] = LabelEncoder().fit_transform(X[column])

# Escalar características
X = StandardScaler().fit_transform(X)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar un modelo SVM lineal
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones y calcular la precisión
y_pred = model.predict(X_test)
precision_linear = accuracy_score(y_test, y_pred)
print("Precisión con kernel 'linear':", precision_linear)

# Cambiar el kernel a 'rbf' y observar cómo afecta el rendimiento
print("\nResultados con kernel 'rbf':")
model_rbf = SVC(kernel='rbf', random_state=42)
model_rbf.fit(X_train, y_train)
y_pred_rbf = model_rbf.predict(X_test)
precision_rbf = accuracy_score(y_test, y_pred_rbf)
print("Precisión con kernel 'rbf':", precision_rbf)

# Experimentar con diferentes valores de C y analizar los resultados
valores_C = [0.1, 1, 10, 100]
resultados_C = []
print("\nResultados con kernel 'rbf' y diferentes valores de C:")
for C in valores_C:
    model = SVC(kernel='rbf', C=C, random_state=42)
    model.fit(X_train, y_train)

    # Hacer predicciones y calcular la precisión
    y_pred = model.predict(X_test)
    precision_C = accuracy_score(y_test, y_pred)
    resultados_C.append((C, precision_C))
    print(f"Precisión con kernel 'rbf' y C={C}: {precision_C}")

# Guardar los resultados en el README.md
readme_path = 'MAquinas de Soporte Ventorial/README.md'
os.makedirs(os.path.dirname(readme_path), exist_ok=True)

with open(readme_path, 'w', encoding='utf-8') as f:
    # Encabezado
    f.write("# Análisis de Máquina de Soporte Vectorial\n\n")
    
    # Descripción del dataset
    f.write("## Descripción del Dataset\n")
    f.write("El dataset utilizado contiene información sobre votos en diferentes distritos. Los atributos incluyen:\n")
    f.write("- UBIGEO: Código único de ubicación geográfica\n")
    f.write("- DEPARTAMENTO: Nombre del departamento\n")
    f.write("- PROVINCIA: Nombre de la provincia\n")
    f.write("- DISTRITO: Nombre del distrito\n")
    f.write("- AUTORIDAD EN CONSULTA: Nombre de la autoridad en consulta\n")
    f.write("- ELECTORES: Número de electores\n")
    f.write("- VOTOS SI: Número de votos a favor\n")
    f.write("- VOTOS NO: Número de votos en contra\n")
    f.write("- VOTOS BLANCOS: Número de votos en blanco\n")
    f.write("- VOTOS NULOS: Número de votos nulos\n")
    f.write("- VOTOS IMPUGNADOS: Número de votos impugnados\n")
    f.write("- VOTOS TOTAL: Número total de votos\n\n")
    
    # Resultados con kernel lineal
    f.write("## Resultados con kernel 'linear'\n")
    f.write(f"Precisión: {precision_linear:.4f}\n")
    f.write("La precisión indica el porcentaje de predicciones correctas realizadas por el modelo con un kernel lineal.\n\n")
    
    # Resultados con kernel RBF
    f.write("## Resultados con kernel 'rbf'\n")
    f.write(f"Precisión: {precision_rbf:.4f}\n")
    f.write("La precisión indica el porcentaje de predicciones correctas realizadas por el modelo con un kernel RBF.\n\n")
    
    # Resultados con kernel RBF y diferentes valores de C
    f.write("## Resultados con kernel 'rbf' y diferentes valores de C\n")
    for C, precision_C in resultados_C:
        f.write(f"Precisión con kernel 'rbf' y C={C}: {precision_C:.4f}\n")
    f.write("La precisión indica el porcentaje de predicciones correctas realizadas por el modelo con un kernel RBF y diferentes valores del parámetro C.\n")
    f.write("El parámetro C controla la penalización de los errores de clasificación. Valores más altos de C intentan clasificar correctamente todos los puntos de entrenamiento, mientras que valores más bajos permiten más errores de clasificación.\n")

print("\nProceso completado exitosamente!")
print(f"Los resultados han sido guardados en {os.path.abspath(readme_path)}")