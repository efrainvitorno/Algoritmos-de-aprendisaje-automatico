import pandas as pd

# Carga el archivo CSV
csv_file = 'archivo.csv'  # Cambia por la ruta de tu archivo CSV
df = pd.read_csv(csv_file)

# Convierte el DataFrame a una tabla Markdown
try:
    from tabulate import tabulate
    markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
except ImportError:
    raise ImportError("Por favor instala la biblioteca 'tabulate' con: pip install tabulate")

# Guarda la tabla Markdown en el README.md
with open('README.md', 'a') as f:  # Modo 'a' para agregar al final del archivo existente
    f.write("\n## Contenido del archivo CSV\n\n")  # Agrega un título para la tabla
    f.write(markdown_table + "\n")

print("Contenido del archivo CSV agregado al README.md")
