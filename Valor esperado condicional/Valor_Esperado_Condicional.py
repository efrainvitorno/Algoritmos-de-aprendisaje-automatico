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

# Guardamos el resultado
resultados += f"### VEC de VOTOS TOTAL dado ELECTORES > 1000\n"
resultados += f"Valor esperado condicional de VOTOS TOTAL dado ELECTORES > 1000: {valor_esperado_condicional_votos_total:.2f}\n\n"

# --------------------------------------------------------------------------------
# Cálculo 2: VEC de VOTOS NULOS dado VOTOS TOTAL > 500
# --------------------------------------------------------------------------------
condicion_votos_nulos = df[df['VOTOS TOTAL'] > 500]['VOTOS NULOS']
valor_esperado_condicional_votos_nulos = condicion_votos_nulos.mean()

# Guardamos el resultado
resultados += f"### VEC de VOTOS NULOS dado VOTOS TOTAL > 500\n"
resultados += f"Valor esperado condicional de VOTOS NULOS dado VOTOS TOTAL > 500: {valor_esperado_condicional_votos_nulos:.2f}\n\n"

# --------------------------------------------------------------------------------
# Cálculo 3: Comparación de VEC de VOTOS BLANCOS para diferentes rangos de ELECTORES
# --------------------------------------------------------------------------------
# VEC para ELECTORES <= 1000
condicion_menor_1000 = df[df['ELECTORES'] <= 1000]['VOTOS BLANCOS']
valor_esperado_menor_1000 = condicion_menor_1000.mean()

# VEC para ELECTORES > 1000
condicion_mayor_1000 = df[df['ELECTORES'] > 1000]['VOTOS BLANCOS']
valor_esperado_mayor_1000 = condicion_mayor_1000.mean()

# Guardamos el resultado
resultados += f"### VEC de VOTOS BLANCOS dado rangos de ELECTORES\n"
resultados += f"VEC de VOTOS BLANCOS dado ELECTORES <= 1000: {valor_esperado_menor_1000:.2f}\n"
resultados += f"VEC de VOTOS BLANCOS dado ELECTORES > 1000: {valor_esperado_mayor_1000:.2f}\n\n"

# --------------------------------------------------------------------------------
# Guardar los resultados y las predicciones en un archivo .md
# --------------------------------------------------------------------------------
with open("Valor esperado condicional/README.md", "w") as f:
    f.write(resultados)

# --------------------------------------------------------------------------------
# Imprimir los resultados
# --------------------------------------------------------------------------------
print(resultados)