import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos desde un archivo CSV (delimitador ";")
df = pd.read_csv("archivo.csv", delimiter=";")

# Realizar los análisis y cálculos necesarios
# Agrupar por DEPARTAMENTO y sumar los votos
resultados_analisis = df.groupby(['DEPARTAMENTO']).agg(
    {'VOTOS SI': 'sum', 'VOTOS NO': 'sum', 'VOTOS BLANCOS': 'sum', 'VOTOS NULOS': 'sum', 'VOTOS TOTAL': 'sum'}).reset_index()

# Crear un gráfico de votos apilados por departamento
fig, ax = plt.subplots(figsize=(12, 8))

# Definir los valores para el gráfico de votos apilados
votos_apilados = resultados_analisis[['VOTOS SI', 'VOTOS NO', 'VOTOS BLANCOS', 'VOTOS NULOS']].set_index(resultados_analisis['DEPARTAMENTO'])
votos_apilados.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')

# Añadir títulos y etiquetas
ax.set_title('Distribución de Votos por Departamento', fontsize=16)
ax.set_xlabel('Departamento', fontsize=12)
ax.set_ylabel('Número de Votos', fontsize=12)

# Guardar el gráfico como una imagen en la carpeta 'Regularizacion/resultados'
plt.tight_layout()  # Ajusta el gráfico para evitar que se corten etiquetas
plt.savefig('votos_apilados.png')
plt.close()

# Crear el contenido en formato Markdown para el archivo README.md
resultados = """
# Resultados del Análisis

## Análisis de Votos

A continuación se muestran los resultados del análisis de los votos por departamento.

### Resumen de Votos

| Departamento | Votos A favor | Votos en Contra | Votos Blancos | Votos Nulos | Total de Votos |
|--------------|---------------|-----------------|---------------|-------------|----------------|
"""

# Añadir los resultados de la tabla por departamento
for _, row in resultados_analisis.iterrows():
    resultados += f"| {row['DEPARTAMENTO']} | {row['VOTOS SI']} | {row['VOTOS NO']} | {row['VOTOS BLANCOS']} | {row['VOTOS NULOS']} | {row['VOTOS TOTAL']} |\n"

# Agregar el gráfico de votos apilados
resultados += """
### Gráfico de Votos Apilados

A continuación se presenta un gráfico con la distribución de votos por departamento.

![Gráfico de Votos Apilados](Regularizacion/votos_apilados.png)

## Conclusión

Los resultados muestran un panorama interesante de la distribución de votos en diversos departamentos. Se observa que algunos departamentos tienen una distribución más balanceada entre los votos a favor y en contra, mientras que otros tienen una diferencia más pronunciada.
"""

# Guardar los resultados en el archivo README.md
with open("Regularizacion/README.md", "w") as f:
    f.write(resultados)

print("Análisis y resultados guardados correctamente en 'README.md'.")
