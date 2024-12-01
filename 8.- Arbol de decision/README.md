# Árbol de Decisión para la Revocatoria

### Descripción del Proyecto
Este proyecto utiliza un modelo de Árbol de Decisión para analizar los resultados distritales de una revocatoria. El objetivo principal es determinar si la cantidad de votos a favor ("VOTOS SI") supera a los votos en contra ("VOTOS NO") en base a características relevantes, como el número de electores, votos blancos, votos nulos, entre otros.

El árbol generado clasifica los distritos en dos categorías:
- **Revocatoria** (cuando "VOTOS SI" > "VOTOS NO").
- **No Revocatoria** (en caso contrario).

---

### Archivos incluidos
- **`Arbol_de_Decision.png`**: Representación gráfica del árbol de decisión entrenado.
- **`modelo_arbol_decision.pkl`**: Modelo entrenado en formato pickle, listo para ser usado en predicciones.
- **`Resultados por distrito de la Revocatoria_Distrital.csv`**: Dataset utilizado para el análisis y entrenamiento del modelo.
- **Código fuente**: Script en Python para la carga de datos, entrenamiento del modelo, y generación del árbol de decisión.

---

### Árbol de Decisión
![Arbol de decision](./Arbol_de_Decision.png)

---
