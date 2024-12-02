import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import datasets

# Función para entrenar y evaluar el modelo SVM
def entrenar_y_evaluar(X, y, dataset_name):
    resultados = []
    # Dividir el conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entrenar un modelo SVM con kernel RBF y diferentes valores de gamma
    for gamma_value in [0.1, 1, 10]:
        model = SVC(kernel='rbf', C=1, gamma=gamma_value, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        resultados.append(f"| SVM Modelo con rbf | rbf | {gamma_value} | {precision:.2f} |")
    
    return resultados

# Cargar el dataset desde archivo.csv con el delimitador correcto
data = pd.read_csv('archivo.csv', delimiter=';')

# Imprimir las primeras filas del DataFrame para verificar los datos
print(data.head())

# Verificar si el DataFrame está vacío
if data.empty:
    raise ValueError("El archivo CSV está vacío o mal formateado.")

# Convertir las columnas categóricas a variables dummy
data = pd.get_dummies(data)

# Asumiendo que la última columna es la etiqueta (target)
X_csv = data.iloc[:, :-1].values
y_csv = data.iloc[:, -1].values

# Verificar las dimensiones de X_csv e y_csv
print(f"Dimensiones de X_csv: {X_csv.shape}")
print(f"Dimensiones de y_csv: {y_csv.shape}")

# Verificar si X_csv tiene características
if X_csv.shape[1] == 0:
    raise ValueError("El array X_csv no tiene características (columnas). Verifica el archivo CSV.")

# Evaluar el dataset cargado desde archivo.csv
resultados = entrenar_y_evaluar(X_csv, y_csv, "archivo.csv")

# Probar con otros datasets disponibles en sklearn
datasets_to_try = {
    "iris": datasets.load_iris(),
    "wine": datasets.load_wine(),
    "digits": datasets.load_digits()
}

for name, dataset in datasets_to_try.items():
    X, y = dataset.data, dataset.target
    resultados.extend(entrenar_y_evaluar(X, y, name))

# Guardar los resultados en el archivo README.md en formato de tabla
with open('9.- MaquinaSoporteVectorial/README.md', 'a', encoding='utf-8') as f:
    f.write("\n## Resultados del Ejercicio 3: Clasificación multiclase con SVM\n")
    f.write("Extender el uso de SVM a problemas con más de dos clases. Experimenta con diferentes valores de gamma y observa cómo afectan los resultados.\n")
    f.write("Prueba con otros datasets disponibles en sklearn, como wine o digits.\n")
    f.write("\n| Modelo | Kernel | Gamma | Precisión |\n")
    f.write("|--------|--------|-------|-----------|\n")
    for resultado in resultados:
        f.write(resultado + "\n")