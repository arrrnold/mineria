# Importación de bibliotecas
import pandas as pd  # Importa la biblioteca Pandas para el manejo de datos
import factor_analyzer as FactorAnalyzer  # Importa FactorAnalyzer para análisis factorial

# Creación de un DataFrame con variables ficticias
df = pd.DataFrame({
    'v1': [6, 4, 9, 1, 2, 5, 3, 0, 4, 1],
    'v2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    'v3': [7, 5, 8, 3, 4, 1, 9, 0, 6, 7],
    'v4': [7, 8, 5, 9, 1, 0, 7, 6, 2, 4],
    'v5': [1, 2, 3, 9, 0, 7, 8, 4, 8, 9],
    'v6': [0, 9, 1, 3, 9, 8, 0, 4, 8, 2],
    'v7': [9, 8, 3, 2, 4, 9, 8, 0, 8, 9]
})

# Imprimir el DataFrame para visualizar los datos
print(df)

# Creación de una instancia del objeto FactorAnalyzer
fa = FactorAnalyzer.FactorAnalyzer(rotation=None)
# Ajuste del análisis factorial exploratorio al DataFrame
fa.fit(df)

# Obtención de las cargas factoriales de las variables en los factores
fa.loadings_

# Obtención de las comunalidades que representan la proporción de varianza explicada por los factores
fa.get_communalities()

# Obtención de los valores propios (eigenvalues) que indican la varianza explicada por cada factor
fa.get_eigenvalues()

# Obtención de información sobre la varianza explicada por los factores
fa.get_factor_variance()

# Obtención de las unicidades que representan la proporción de varianza no explicada por los factores
fa.get_uniquenesses()
