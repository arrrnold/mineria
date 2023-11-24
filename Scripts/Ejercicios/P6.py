import pandas as pd

# Cargar los datos desde un csv
df = pd.read_csv("../../Datasets/data_ejemplo06.csv")

# ver los primeros datos de DF
print("Datos originales")
print(df)
print()

# Agregar ventas por producto
print("Ventas totales por producto: ")
ventas_por_producto = df.groupby('Producto'['Ventas'].sum())
print(ventas_por_producto)
print()

# Agregar ventas por region
print("Ventas totales por region: ")
ventas_por_region = df.groupby('Region'['Ventas'].sum())
print(ventas_por_region)
print()

# Agregar ventas por producto y region
print("Ventas totales por producto y region: ")
ventas_por_producto_y_region = df.groupby(['Producto', 'Region']) ['Ventas'].sum().unstack() # unstack para ordenar los datos
print(ventas_por_producto_y_region)