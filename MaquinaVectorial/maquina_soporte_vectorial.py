import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

class TestSVMArchivoCsv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuración inicial"""
        try:
            # Asegurar que el directorio existe
            os.makedirs('MaquinaVectorial', exist_ok=True)
            
            cls.data = pd.read_csv('archivo.csv', delimiter=';')
            cls.X = cls.data[['ELECTORES', 'VOTOS SI', 'VOTOS NO', 'VOTOS BLANCOS', 'VOTOS NULOS', 'VOTOS TOTAL']].values
            cls.y = (cls.data['VOTOS SI'] > cls.data['VOTOS NO']).astype(int)
            cls.scaler = StandardScaler()
            cls.X_scaled = cls.scaler.fit_transform(cls.X)
            cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
                cls.X_scaled, cls.y, test_size=0.3, random_state=42
            )
            cls.resultados = []
            print("Configuración inicial completada exitosamente")
        except Exception as e:
            print(f"Error en la configuración: {str(e)}")
            raise

    def test_svm_lineal(self):
        """Probar SVM con kernel lineal"""
        try:
            model = SVC(kernel='linear', random_state=42)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.resultados.append({
                'nombre': 'SVM con Kernel Lineal',
                'accuracy': accuracy
            })
            print(f"Prueba kernel lineal completada. Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Error en prueba lineal: {str(e)}")
            raise

    def test_svm_rbf(self):
        """Probar SVM con kernel RBF"""
        try:
            model = SVC(kernel='rbf', random_state=42)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.resultados.append({
                'nombre': 'SVM con Kernel RBF',
                'accuracy': accuracy
            })
            print(f"Prueba kernel RBF completada. Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Error en prueba RBF: {str(e)}")
            raise

    def guardar_resultados(self):
        """Guardar resultados en README.md"""
        try:
            readme_path = 'MaquinaVectorial/README.md'
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("# Resultados de Pruebas de Máquina de Soporte Vectorial\n\n")
                f.write(f"## Descripción del Dataset\n")
                f.write(f"Total de registros: {len(self.data)}\n\n")
                f.write("## Resultados\n\n")
                for resultado in self.resultados:
                    f.write(f"### {resultado['nombre']}\n")
                    f.write(f"Precisión: {resultado['accuracy']:.4f}\n\n")
            print(f"Resultados guardados en {os.path.abspath(readme_path)}")
        except Exception as e:
            print(f"Error guardando resultados: {str(e)}")
            raise

    def tearDown(self):
        """Guardar resultados después de cada prueba"""
        self.guardar_resultados()

if __name__ == '__main__':
    unittest.main(verbosity=2)