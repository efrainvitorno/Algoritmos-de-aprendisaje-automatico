import pandas as pd

# Cargar los datos desde un archivo CSV (delimitador ";")
df = pd.read_csv("archivo.csv", delimiter=";")

# Variable para almacenar todos los resultados
resultados = ""

# --------------------------------------------------------------------------------
# Concepto de Valor Esperado Condicional
# --------------------------------------------------------------------------------
resultados += (
    "### ¿Qué es el Valor Esperado Condicional?\n\n"
    "El **Valor Esperado Condicional** (VEC) es una medida estadística que describe la expectativa o el promedio de "
    "una variable aleatoria, dado que se cumple una condición específica sobre otra variable. En términos más simples, "
    "es el valor promedio de una variable cuando conocemos algo sobre otra. El VEC se usa en muchos campos como la "
    "economía, las ciencias sociales y las ciencias políticas para analizar cómo cambia una variable bajo ciertas "
    "circunstancias.\n\n"
)

# --------------------------------------------------------------------------------
# Cálculo 1: VEC de VOTOS TOTAL dado ELECTORES > 1000
# --------------------------------------------------------------------------------
condicion_votos_total = df[df['ELECTORES'] > 1000]['VOTOS TOTAL']
valor_esperado_condicional_votos_total = condicion_votos_total.mean()

# Guardamos el resultado con explicación detallada
resultados += f"### VEC de VOTOS TOTAL dado ELECTORES > 1000\n"
resultados += (
    f"- **Descripción del cálculo**: Este cálculo muestra el **Valor Esperado Condicional** de los **VOTOS TOTAL** en "
    f"las localidades donde el número de **ELECTORES** supera los 1000. Es decir, estamos analizando el comportamiento de "
    f"los votos totales en aquellas áreas con una mayor población electoral.\n\n"
    f"- **Resultado**: El VEC de VOTOS TOTALES es de {valor_esperado_condicional_votos_total:.2f}. Esto significa que, en promedio, en "
    f"las localidades con más de 1000 electores, el número total de votos es de aproximadamente {valor_esperado_condicional_votos_total:.2f}.\n\n"
    f"- **Interpretación**: Este valor puede ser útil para entender las dinámicas de votación en áreas con mayor densidad electoral. "
    f"Un número mayor de electores podría reflejar un mayor interés o una mayor representación política, lo que se puede traducir "
    f"en un aumento de la participación electoral y en el número de votos emitidos.\n\n"
)

# --------------------------------------------------------------------------------
# Cálculo 2: VEC de VOTOS NULOS dado VOTOS TOTAL > 500
# --------------------------------------------------------------------------------
condicion_votos_nulos = df[df['VOTOS TOTAL'] > 500]['VOTOS NULOS']
valor_esperado_condicional_votos_nulos = condicion_votos_nulos.mean()

# Guardamos el resultado con explicación detallada
resultados += f"### VEC de VOTOS NULOS dado VOTOS TOTAL > 500\n"
resultados += (
    f"- **Descripción del cálculo**: Este cálculo nos muestra el **Valor Esperado Condicional** de los **VOTOS NULOS**, dado que "
    f"el total de **VOTOS TOTAL** supera los 500. Este análisis ayuda a entender cómo la cantidad de votos nulos varía en relación "
    f"a un número elevado de votos emitidos.\n\n"
    f"- **Resultado**: El VEC de VOTOS NULOS es de {valor_esperado_condicional_votos_nulos:.2f}. Esto indica que, en promedio, en las localidades "
    f"donde los votos totales superan los 500, el número de votos nulos es de aproximadamente {valor_esperado_condicional_votos_nulos:.2f}.\n\n"
    f"- **Interpretación**: Este valor es importante para entender los patrones de rechazo hacia los candidatos o la falta de confianza en "
    f"el proceso electoral. Un mayor número de votos nulos podría reflejar desinformación, frustración o desinterés hacia el sistema electoral.\n\n"
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
    f"- **Descripción del cálculo**: Este análisis compara el **Valor Esperado Condicional** de los **VOTOS BLANCOS** en dos grupos: "
    f"las localidades con **ELECTORES <= 1000** y las que tienen **ELECTORES > 1000**. De esta forma, podemos observar cómo varía la "
    f"proporción de votos blancos según el tamaño del electorado.\n\n"
    f"- **Resultado**: El VEC de VOTOS BLANCOS para **ELECTORES <= 1000** es de {valor_esperado_menor_1000:.2f}, mientras que el VEC de VOTOS BLANCOSpara **ELECTORES > 1000** es de {valor_esperado_mayor_1000:.2f}.\n\n"
    f"- **Interpretación**: Estos valores muestran que, en promedio, las localidades con menos de 1000 electores tienen un número distinto "
    f"de votos blancos comparado con aquellas que tienen más de 1000 electores. Esto puede ser indicativo de patrones de desinterés, "
    f"desconfianza en los candidatos o falta de información en diferentes contextos electorales. \n\n"
    f"El cambio en el número de votos blancos según el tamaño del electorado puede revelar distintas actitudes políticas o comportamientos en las urnas.\n\n"
)

# --------------------------------------------------------------------------------
# Guardar los resultados y las predicciones en un archivo .md
# --------------------------------------------------------------------------------
with open("Valor esperado condicional/README.md", "w") as f:
    f.write(resultados)

