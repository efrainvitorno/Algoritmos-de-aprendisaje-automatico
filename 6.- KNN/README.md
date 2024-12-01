# Análisis de la Revocatoria

Este proyecto analiza los resultados de la revocatoria distrital utilizando un modelo de KNN.

## Resultados

### Matriz de Confusión
![Matriz de Confusión](./matriz_confusion.png)

### Gráfico de Precisión
![Gráfico de Precisión](./KNN_precision_vs_K.png)
## Análisis de Precisión para Diferentes Valores de K

El valor de **K** que proporciona la mayor precisión es **K = 2**, donde la precisión alcanza su punto máximo de aproximadamente **0.92**.

A medida que **K** aumenta, la precisión tiende a disminuir gradualmente después de alcanzar su pico. Esto podría deberse a que el modelo se vuelve menos sensible a los datos locales al considerar más vecinos, lo que reduce su capacidad de ajustar los patrones de los datos de entrenamiento.

### Gráfico de Precisión vs. K:

Se puede observar que al aumentar el valor de **K**, el modelo se estabiliza, pero su capacidad para ajustarse a los datos locales disminuye.

| K   | Precisión |
|-----|-----------|
| 1   | 0.90      |
| 2   | 0.92      |
| 3   | 0.91      |
| 4   | 0.89      |
| 5   | 0.87      |
| 6   | 0.85      |

### Conclusión

El modelo muestra un comportamiento óptimo para **K = 2**, y a medida que **K** crece, el modelo se vuelve más general, perdiendo algo de precisión. Es importante elegir un valor de **K** que balancee la complejidad del modelo y la capacidad de generalización para obtener mejores resultados.



## Reporte de Clasificación KNN

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| **0** | 90%       | 83%    | 86%      | 42      |
| **1** | 95%       | 97%    | 96%      | 145     |

### Métricas Generales:

- **Precisión Total (Accuracy):** 94%
- **Promedio Macro (Macro avg):**
  - **Precisión:** 93%
  - **Recall:** 90%
  - **F1-Score:** 91%
- **Promedio Ponderado (Weighted avg):**
  - **Precisión:** 94%
  - **Recall:** 94%
  - **F1-Score:** 94%

 