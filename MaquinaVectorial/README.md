# Proyecto Maquina Vectorial
Este proyecto explora la clasificaci�n utilizando M�quinas de Vectores de Soporte (SVM) con diferentes kernels y par�metros.

## Ejercicio 1: Clasificaci�n SVM con el dataset Iris
Precisi�n con kernel 'linear': 1.0000
Precisi�n con kernel 'rbf': 1.0000
### Resultados con kernel 'rbf' y diferentes valores de C:
Precisi�n con kernel 'rbf' y C=0.1: 1.0000
Precisi�n con kernel 'rbf' y C=1: 1.0000
Precisi�n con kernel 'rbf' y C=10: 1.0000
Precisi�n con kernel 'rbf' y C=100: 1.0000

---
## Ejercicio 2: Clasificaci�n SVM con Datos Sint�ticos
### L�mites de decisi�n con kernel 'linear':
![L�mites de decisi�n con kernel 'linear'](limites_lineal.png)
### L�mites de decisi�n con kernel 'rbf':
![L�mites de decisi�n con kernel 'rbf'](Malimites_rbf.png)
### L�mites de decisi�n con kernel 'poly':
![L�mites de decisi�n con kernel 'poly'](limites_poly.png)
### L�mites de decisi�n con kernel 'rbf' y C=0.1:
![L�mites de decisi�n con kernel 'rbf' y C=0.1](limites_rbf_C0.1.png)
### L�mites de decisi�n con kernel 'rbf' y C=1:
![L�mites de decisi�n con kernel 'rbf' y C=1](limites_rbf_C1.png)
### L�mites de decisi�n con kernel 'rbf' y C=10:
![L�mites de decisi�n con kernel 'rbf' y C=10](limites_rbf_C10.png)

---
### Conclusi�n
Se observ� que la precisi�n de los modelos SVM depende del kernel y del par�metro C. El modelo con kernel 'rbf' mostr� un buen rendimiento en ambos ejercicios. Las gr�ficas generadas muestran claramente los l�mites de decisi�n para cada tipo de kernel.
