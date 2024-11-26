# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os

try:
    # Cargar y preparar datos
    file_path = 'archivo.csv'
    data = pd.read_csv(file_path, sep=';')
    
    # Preprocesamiento
    X = data.drop(columns=['VOTOS TOTAL'])
    y = data['VOTOS TOTAL']

    # Codificar variables categóricas
    for column in X.select_dtypes(include=['object']).columns:
        X[column] = LabelEncoder().fit_transform(X[column])

    # Escalar características
    X = StandardScaler().fit_transform(X)

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entrenar modelo
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)

    # Predicciones
    y_pred = svm_model.predict(X_test)

    # Métricas
    classification_report_str = classification_report(y_test, y_pred)
    accuracy_score_str = f"Exactitud del modelo: {accuracy_score(y_test, y_pred):.4f}"
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()
    plt.ylabel('Verdaderos')
    plt.xlabel('Predicciones')
    
    # Agregar valores numéricos
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    # Guardar visualización
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Modificar la sección de escritura del README:
    readme_path = 'MAquinas de Soporte Ventorial/README.md'
    os.makedirs(os.path.dirname(readme_path), exist_ok=True)

    with open(readme_path, 'w', encoding='utf-8') as f:
        # Encabezado
        f.write("# Análisis de Máquina de Soporte Vectorial\n\n")
        
        # Descripción del dataset
        f.write("## Descripción del Dataset\n")
        f.write(f"Total de registros: {len(data)}\n\n")
        
        # Resultados con kernel lineal
        f.write("## Resultados con kernel 'linear'\n")
        model_linear = SVC(kernel='linear', random_state=42)
        model_linear.fit(X_train, y_train)
        y_pred_linear = model_linear.predict(X_test)
        accuracy_linear = accuracy_score(y_test, y_pred_linear)
        f.write(f"Precisión: {accuracy_linear:.4f}\n\n")
        
        # Resultados con kernel RBF
        f.write("## Resultados con kernel 'rbf'\n")
        model_rbf = SVC(kernel='rbf', random_state=42)
        model_rbf.fit(X_train, y_train)
        y_pred_rbf = model_rbf.predict(X_test)
        accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
        f.write(f"Precisión: {accuracy_rbf:.4f}\n\n")

    print("\nProceso completado exitosamente!")
    print(f"Los resultados han sido guardados en {os.path.abspath(readme_path)}")

except FileNotFoundError as e:
    print(f"Error: No se pudo encontrar el archivo: {str(e)}")
except PermissionError as e:
    print(f"Error: No hay permisos para escribir el archivo: {str(e)}")
except Exception as e:
    print(f"Error inesperado: {str(e)}")
finally:
    plt.close('all')