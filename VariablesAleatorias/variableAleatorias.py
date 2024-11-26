import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Cargar el dataset desde el archivo local
url = "archivo.csv"
df = pd.read_csv(url, delimiter=';')

# Verificar las columnas necesarias
if 'VOTOS TOTAL' not in df.columns or 'ELECTORES' not in df.columns:
    raise ValueError("El dataset debe contener las columnas 'VOTOS TOTAL' y 'ELECTORES'.")

# Calcular estadísticas básicas
votos = df['VOTOS TOTAL'].dropna()
promedio_x = votos.mean()
total_electores = df['ELECTORES'].sum()
total_asistieron = votos.sum()
total_no_asistieron = total_electores - total_asistieron

# Crear tabla para el resumen general
resumen_general = f"""
| Descripción                                | Valor      |
|--------------------------------------------|------------|
| Total de ciudadanos que asistieron a votar | {total_asistieron} |
| Total de ciudadanos que no asistieron a votar | {total_no_asistieron} |
| Promedio de los votos en la población      | {promedio_x:.2f} |
"""

# Ejercicio 2: Diferencia entre promedio de población y muestra (semilla 1)
np.random.seed(1)
muestra = np.random.choice(votos, size=5, replace=False)
promedio_muestra = np.mean(muestra)
diferencia_abs = abs(promedio_muestra - promedio_x)

# Ejercicio 3: Diferencia con semilla 5
np.random.seed(5)
muestra_2 = np.random.choice(votos, size=5, replace=False)
promedio_muestra_2 = np.mean(muestra_2)
diferencia_abs_2 = abs(promedio_muestra_2 - promedio_x)

# Crear tabla para las diferencias
tabla_diferencias = f"""
| Ejercicio | Promedio Muestra | Diferencia Absoluta |
|-----------|------------------|---------------------|
| Semilla 1 | {promedio_muestra:.2f}          | {diferencia_abs:.2f}            |
| Semilla 5 | {promedio_muestra_2:.2f}          | {diferencia_abs_2:.2f}            |
"""

# Ejercicio 4 y 5: Simulaciones
np.random.seed(1)
promedios_1000 = [np.mean(np.random.choice(votos, size=5, replace=False)) for _ in range(1000)]
porcentaje_1000 = np.sum(np.abs(promedios_1000 - promedio_x) > 1) / 1000 * 100

promedios_10k = [np.mean(np.random.choice(votos, size=5, replace=False)) for _ in range(10000)]
porcentaje_10k = np.sum(np.abs(promedios_10k - promedio_x) > 1) / 10000 * 100

tabla_simulaciones = f"""
| Simulación      | Porcentaje a más de 1 del promedio |
|-----------------|------------------------------------|
| 1000 muestras   | {porcentaje_1000:.2f}%            |
| 10,000 muestras | {porcentaje_10k:.2f}%            |
"""

# Ejercicio 6: Simulaciones con muestra de tamaño 50
promedios_50 = [np.mean(np.random.choice(votos, size=50, replace=False)) for _ in range(1000)]
porcentaje_50 = np.sum(np.abs(promedios_50 - promedio_x) > 1) / 1000 * 100

# Crear el histograma y guardarlo
plt.figure(figsize=(12, 6))
plt.hist(promedios_1000, bins=30, alpha=0.5, label="Tamaño muestra 5")
plt.hist(promedios_50, bins=30, alpha=0.5, label="Tamaño muestra 50")
plt.legend()
plt.xlabel("Promedios")
plt.ylabel("Frecuencia")
plt.title("Comparación de histogramas (muestra 5 vs 50)")
plt.savefig('VariablesAleatorias/histogramas.png')
plt.close()

# Ejercicio 8: Porcentaje en rango 23-25
rango = (23, 25)
promedios_50_array = np.array(promedios_50)
porcentaje_rango = np.sum((promedios_50_array >= rango[0]) & (promedios_50_array <= rango[1])) / len(promedios_50_array) * 100

# Comparación con distribución normal
mu, sigma = 23.9, 0.43
prob_normal = norm.cdf(rango[1], mu, sigma) - norm.cdf(rango[0], mu, sigma)

tabla_rango = f"""
| Rango  | Porcentaje Promedios (Muestra 50) | Porcentaje Esperado (Normal) |
|--------|-----------------------------------|-------------------------------|
| 23-25  | {porcentaje_rango:.2f}%                      | {prob_normal * 100:.2f}%                     |
"""

# Escribir resultados en README.md
with open("VariablesAleatorias/README.md", "w", encoding="utf-8") as f:
    f.write("# Resultados de Análisis de Votos\n\n")
    f.write("## Resumen General\n")
    f.write(resumen_general)
    f.write("\n## Ejercicio 2 y 3: Diferencias de Promedio\n")
    f.write(tabla_diferencias)
    f.write("\n## Ejercicio 4 y 5: Resultados de Simulaciones\n")
    f.write(tabla_simulaciones)
    f.write("\n## Ejercicio 8: Porcentaje en Rango 23-25\n")
    f.write(tabla_rango)
    f.write("\n## Comparación de Histogramas\n")
    f.write("![Comparación de histogramas](histogramas.png)\n")
