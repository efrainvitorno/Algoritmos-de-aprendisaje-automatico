# Análisis de Máquina de Soporte Vectorial

## Descripción del Dataset
El dataset utilizado contiene información sobre votos en diferentes distritos. Los atributos incluyen:
- UBIGEO: Código único de ubicación geográfica
- DEPARTAMENTO: Nombre del departamento
- PROVINCIA: Nombre de la provincia
- DISTRITO: Nombre del distrito
- AUTORIDAD EN CONSULTA: Nombre de la autoridad en consulta
- ELECTORES: Número de electores
- VOTOS SI: Número de votos a favor
- VOTOS NO: Número de votos en contra
- VOTOS BLANCOS: Número de votos en blanco
- VOTOS NULOS: Número de votos nulos
- VOTOS IMPUGNADOS: Número de votos impugnados
- VOTOS TOTAL: Número total de votos

## Resultados con kernel 'linear'
Precisión: 0.5882

## Resultados con kernel 'rbf'
Precisión: 0.2620

## Resultados con kernel 'rbf' y diferentes valores de C
Precisión con kernel 'rbf' y C=0.1: 0.0000
Precisión con kernel 'rbf' y C=1: 0.2620
Precisión con kernel 'rbf' y C=10: 0.6417
Precisión con kernel 'rbf' y C=100: 0.6364
