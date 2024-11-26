import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Cargar el dataset desde el archivo local
url = "archivo.csv"
df = pd.read_csv(url, delimiter=';')

# Imprimir las columnas del DataFrame para verificar
print("Columnas del DataFrame:", df.columns)

# Verificar si la columna 'VOTOS TOTAL' existe en el DataFrame
if 'VOTOS TOTAL' not in df.columns:
    raise ValueError("La columna 'VOTOS TOTAL' no existe en el dataset")

# Asegurarse de que la columna que contiene los votos se llama 'VOTOS TOTAL'
votos = df['VOTOS TOTAL'].dropna()  # Elimina valores nulos si existen
promedio_x = votos.mean()

# Guardar resultados en una lista
resultados = []

resultados.append(f"**Ejercicio 1:** Promedio de los votos en la población: **{promedio_x:.2f}**")

# Ejercicio 2: Diferencia entre promedio de la población y muestra con semilla 1
np.random.seed(1)
muestra = np.random.choice(votos, size=5, replace=False)
promedio_muestra = np.mean(muestra)
diferencia_abs = abs(promedio_muestra - promedio_x)
resultados.append(f"**Ejercicio 2:** Diferencia absoluta (semilla 1): **{diferencia_abs:.2f}**")

# Ejercicio 3: Diferencia entre promedio de la población y muestra con semilla 5
np.random.seed(5)
muestra_2 = np.random.choice(votos, size=5, replace=False)
promedio_muestra_2 = np.mean(muestra_2)
diferencia_abs_2 = abs(promedio_muestra_2 - promedio_x)
resultados.append(f"**Ejercicio 3:** Diferencia absoluta (semilla 5): **{diferencia_abs_2:.2f}**")
resultados.append("**Respuesta:** C) Porque el promedio de las muestras es una variable aleatoria.")

# Ejercicio 4: 1000 simulaciones de tamaño de muestra 5
np.random.seed(1)
promedios_1000 = [np.mean(np.random.choice(votos, size=5, replace=False)) for _ in range(1000)]
porcentaje_1000 = np.sum(np.abs(promedios_1000 - promedio_x) > 1) / 1000 * 100
resultados.append(f"**Ejercicio 4:** Porcentaje de promedios a más de 1 del promedio de x (1000 muestras): **{porcentaje_1000:.2f}%**")

# Ejercicio 5: 10,000 simulaciones de tamaño de muestra 5
np.random.seed(1)
promedios_10k = [np.mean(np.random.choice(votos, size=5, replace=False)) for _ in range(10000)]
porcentaje_10k = np.sum(np.abs(promedios_10k - promedio_x) > 1) / 10000 * 100
resultados.append(f"**Ejercicio 5:** Porcentaje de promedios a más de 1 del promedio de x (10,000 muestras): **{porcentaje_10k:.2f}%**")

# Ejercicio 6: 1000 simulaciones de tamaño de muestra 50
np.random.seed(1)
promedios_50 = [np.mean(np.random.choice(votos, size=50, replace=False)) for _ in range(1000)]
porcentaje_50 = np.sum(np.abs(promedios_50 - promedio_x) > 1) / 1000 * 100
resultados.append(f"**Ejercicio 6:** Porcentaje de promedios a más de 1 del promedio de x (tamaño muestra 50): **{porcentaje_50:.2f}%**")

# Ejercicio 7: Comparar histogramas
plt.figure(figsize=(12, 6))
plt.hist(promedios_1000, bins=30, alpha=0.5, label="Tamaño muestra 5")
plt.hist(promedios_50, bins=30, alpha=0.5, label="Tamaño muestra 50")
plt.legend()
plt.xlabel("Promedios")
plt.ylabel("Frecuencia")
plt.title("Comparación de histogramas (muestra 5 vs 50)")
plt.savefig('histogramas.png')
plt.close()
resultados.append("**Respuesta:** B) Ambos se ven más o menos normales, pero con un tamaño de muestra de 50, la dispersión es menor.")

# Ejercicio 8: Porcentaje en rango 23-25
rango = (23, 25)
promedios_50_array = np.array(promedios_50)  # Convertir la lista a un array de NumPy
porcentaje_rango = np.sum((promedios_50_array >= rango[0]) & (promedios_50_array <= rango[1])) / len(promedios_50_array) * 100
resultados.append(f"**Ejercicio 8:** Porcentaje de promedios entre 23 y 25 (muestra 50): **{porcentaje_rango:.2f}%**")

# Comparación con distribución normal
mu, sigma = 23.9, 0.43
prob_normal = norm.cdf(rango[1], mu, sigma) - norm.cdf(rango[0], mu, sigma)
resultados.append(f"**Ejercicio 8:** Porcentaje esperado para distribución normal en rango 23-25: **{prob_normal * 100:.2f}%**")

# Escribir resultados en README.md
with open("README.md", "w") as f:
    f.write("# Resultados de Análisis de Votos\n\n")
    f.write("Este documento presenta los resultados del análisis de los votos totales en el dataset proporcionado.\n\n")
    for resultado in resultados:
        f.write(f"{resultado}\n\n")
    f.write("![Comparación de histogramas](histogramas.png)\n")