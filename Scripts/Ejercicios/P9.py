# P9 Visualización de Diferentes Tipos de Datos en Pandas
"""

# Importar las bibliotecas necesarias
import pandas as pd               # Importar pandas para manipulación de datos
import numpy as np                # Importar numpy para manipulación de arreglos
from datetime import datetime     # Importar datetime para trabajar con fechas

# Datos categóricos
categorias = pd.Series(['rojo', 'azul', 'verde', 'rojo'], dtype='category')  # Crear una Serie de datos categóricos
print("Datos categóricos:")
print(categorias)

# Datos ordinales
ordinales = pd.Categorical(['bajo', 'medio', 'alto', 'medio'], categories=['bajo', 'medio', 'alto'], ordered=True)  # Crear datos ordinales
print("Datos ordinales:")
print(ordinales)

# Datos numéricos
# Continuos
continuos = np.array([25.5, 30.2, 25.7, 40.1])  # Crear un arreglo de datos numéricos continuos
print("Datos continuos:")
print(continuos)

# Datos temporales
fechas = pd.Series([datetime(2023, 1, 1), datetime(2023, 1, 2)])  # Crear una Serie de fechas
print("Datos temporales:")
print(fechas)

# Datos de texto
textos = pd.Series(["Hola mundo", "Minería de datos en Python", "tipo de datos"])  # Crear una Serie de datos de texto
print("Datos de texto:")
print(textos)

# Datos multidimensionales (DataFrame)
data = {
    'categorica': ['A', 'B', 'C', 'D'],
    'numerica': [10, 20, 30, 40]
}

df = pd.DataFrame(data)  # Crear un DataFrame con datos multidimensionales
print("Datos multidimensionales (DataFrame):")
print(df)