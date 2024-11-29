import pandas as pd

# Cargar los datos desde un archivo CSV (delimitador ";")
df = pd.read_csv("archivo.csv", delimiter=";")

# Variable para almacenar todos los resultados
resultados = ""

# --------------------------------------------------------------------------------
# Cálculo 1: VEC de VOTOS TOTAL dado ELECTORES > 1000
# --------------------------------------------------------------------------------
condicion_votos_total = df[df['ELECTORES'] > 1000]['VOTOS TOTAL']
valor_esperado_condicional_votos_total = condicion_votos_total.mean()

# Guardamos el resultado con explicación detallada
resultados += f"### VEC de VOTOS TOTAL dado ELECTORES > 1000\n"
resultados += (
    f"El **Valor Esperado Condicional** (VEC) de **VOTOS TOTAL** dado que el número de **ELECTORES** es mayor a 1000 "
    f"es de {valor_esperado_condicional_votos_total:.2f}. Este valor representa la expectativa del número de votos "
    f"totales en aquellas localidades donde el número de electores supera los 1000. En otras palabras, nos indica "
    f"el promedio de votos totales en los lugares con más de 1000 electores.\n"
    f"Este resultado puede ser útil para entender cómo se comportan los votos en lugares con mayor densidad electoral, "
    f"y puede sugerir tendencias o comportamientos particulares en esos contextos.\n\n"
)

# --------------------------------------------------------------------------------
# Cálculo 2: VEC de VOTOS NULOS dado VOTOS TOTAL > 500
# --------------------------------------------------------------------------------
condicion_votos_nulos = df[df['VOTOS TOTAL'] > 500]['VOTOS NULOS']
valor_esperado_condicional_votos_nulos = condicion_votos_nulos.mean()

# Guardamos el resultado con explicación detallada
resultados += f"### VEC de VOTOS NULOS dado VOTOS TOTAL > 500\n"
resultados += (
    f"El **Valor Esperado Condicional** (VEC) de **VOTOS NULOS** dado que el número de **VOTOS TOTAL** es mayor a 500 "
    f"es de {valor_esperado_condicional_votos_nulos:.2f}. Este cálculo nos da el promedio de votos nulos en aquellas "
    f"localidades donde el total de votos supera los 500. El VEC es una herramienta que permite ver cómo varía el número "
    f"de votos nulos en función de los votos totales registrados.\n"
    f"Este resultado puede reflejar situaciones en las cuales un mayor número de votos totales se asocia a un aumento o "
    f"disminución en los votos nulos, y puede ser útil para identificar patrones de desinformación o rechazo hacia los "
    f"candidatos.\n\n"
)

# --------------------------------------------------------------------------------
# Cálculo 3: Comparación de VEC de VOTOS BLANCOS para diferentes rangos de ELECTORES
# --------------------------------------------------------------------------------
# VEC para ELECTORES <= 1000
condicion_menor_1000 = df[df['ELECTORES'] <= 1000]['VOTOS BLANCOS']
valor_esperado_menor_1000 = condicion_menor_1000.mean()

# VEC para ELECTORES > 1000
condicion_mayor_1000 = df[df['ELECTORES'] > 1000]['VOTOS BLANCOS']
valor_esperado_mayor_1000 = condicion_mayor_1000.mean()

# Guardamos el resultado con explicación detallada
resultados += f"### VEC de VOTOS BLANCOS dado rangos de ELECTORES\n"
resultados += (
    f"El **Valor Esperado Condicional** (VEC) de **VOTOS BLANCOS** se analiza en dos grupos según el número de "
    f"electores. En localidades con **ELECTORES <= 1000**, el VEC de votos blancos es de {valor_esperado_menor_1000:.2f}, "
    f"mientras que en localidades con **ELECTORES > 1000**, el VEC es de {valor_esperado_mayor_1000:.2f}.\n"
    f"Esto nos muestra cómo cambia la proporción de votos blancos dependiendo del tamaño del electorado. Un número mayor de "
    f"electores podría indicar una mayor diversidad de opiniones, lo que puede traducirse en una mayor cantidad de votos "
    f"blancos. Este análisis es importante para entender los patrones de rechazo de candidatos o de desinterés en el proceso "
    f"electoral en función del tamaño de la población electoral.\n\n"
)

# --------------------------------------------------------------------------------
# Guardar los resultados y las predicciones en un archivo .md
# --------------------------------------------------------------------------------
with open("Valor esperado condicional/README.md", "w") as f:
    f.write(resultados)

# --------------------------------------------------------------------------------
# Imprimir los resultados
# --------------------------------------------------------------------------------
print(resultados)
