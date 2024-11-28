# Proyecto Maquina Vectorial
Este proyecto explora la clasificación utilizando Máquinas de Vectores de Soporte (SVM) con diferentes kernels y parámetros.

## Ejercicio 1: Clasificación SVM con el dataset Iris
Precisión con kernel 'linear': 1.0000
Precisión con kernel 'rbf': 1.0000
### Resultados con kernel 'rbf' y diferentes valores de C:
Precisión con kernel 'rbf' y C=0.1: 1.0000
Precisión con kernel 'rbf' y C=1: 1.0000
Precisión con kernel 'rbf' y C=10: 1.0000
Precisión con kernel 'rbf' y C=100: 1.0000

---
## Ejercicio 2: Clasificación SVM con Datos Sintéticos
### Límites de decisión con kernel 'linear':
![Límites de decisión con kernel 'linear'](limites_lineal.png)
### Límites de decisión con kernel 'rbf':
![Límites de decisión con kernel 'rbf'](limites_rbf.png)
### Límites de decisión con kernel 'poly':
![Límites de decisión con kernel 'poly'](limites_poly.png)
### Límites de decisión con kernel 'rbf' y C=0.1:
![Límites de decisión con kernel 'rbf' y C=0.1](limites_rbf_C0.1.png)
### Límites de decisión con kernel 'rbf' y C=1:
![Límites de decisión con kernel 'rbf' y C=1](limites_rbf_C1.png)
### Límites de decisión con kernel 'rbf' y C=10:
![Límites de decisión con kernel 'rbf' y C=10](limites_rbf_C10.png)

---
### Conclusión
Se observó que la precisión de los modelos SVM depende del kernel y del parámetro C. El modelo con kernel 'rbf' mostró un buen rendimiento en ambos ejercicios. Las gráficas generadas muestran claramente los límites de decisión para cada tipo de kernel.
