import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Leer el archivo CSV con separador punto y coma
data = pd.read_csv('archivo.csv', sep=';')  # Reemplaza 'archivo.csv' con el nombre de tu archivo CSV

# Convertir las columnas categóricas a numéricas
label_encoder = LabelEncoder()

# Asegúrate de aplicar el encoder solo a las columnas categóricas (las que son de tipo objeto)
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_encoder.fit_transform(data[col])

# Suponemos que la última columna es la etiqueta y el resto son las características
X = data.iloc[:, :-1].values  # Características (todas las columnas menos la última)
y = data.iloc[:, -1].values   # Etiquetas (última columna)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar un modelo SVM con kernel 'linear'
model_linear = SVC(kernel='linear', random_state=42)
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)
precision_linear = accuracy_score(y_test, y_pred_linear)

# Cambiar el kernel a 'rbf' y entrenar el modelo
model_rbf = SVC(kernel='rbf', random_state=42)
model_rbf.fit(X_train, y_train)
y_pred_rbf = model_rbf.predict(X_test)
precision_rbf = accuracy_score(y_test, y_pred_rbf)

# Experimentar con diferentes valores de C y analizar los resultados
valores_C = [0.1, 1, 10, 100]
resultados_C = {}
for C in valores_C:
    model = SVC(kernel='rbf', C=C, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    resultados_C[C] = accuracy_score(y_test, y_pred)

# Guardar los resultados en el archivo README.md en formato de tabla
with open('MaquinaVectorial/README.md', 'w') as f:
    f.write("# Resultados detallados del modelo SVM\n\n")
    
    # Tabla de resultados con precisión para kernel 'linear' y 'rbf'
    f.write("| **Modelo**                | **Kernel** | **Valor de C** | **Precisión** |\n")
    f.write("|---------------------------|------------|----------------|---------------|\n")
    f.write(f"| SVM Modelo Lineal         | linear     | N/A            | {precision_linear:.2f}         |\n")
    f.write(f"| SVM Modelo con RBF        | rbf        | N/A            | {precision_rbf:.2f}         |\n")

    # Resultados con kernel 'rbf' y diferentes valores de C
    for C, precision in resultados_C.items():
        f.write(f"| SVM Modelo con RBF        | rbf        | {C}            | {precision:.2f}         |\n")
    
    f.write("\nResultados guardados en MaquinaVectorial/README.md")

print("\nResultados guardados en MaquinaVectorial/README.md")
