import pandas as pd # ayuda a hacer analisis de datos

# suponemos que tenemos un conjunto de datos de ventas en un DF
data = {
    'Fecha':    ['2021-01-01','2021-01-01','2021-01-02','2021-01-02'],
    'Producto': ['Manzana', 'Banana', 'Manzana','Banana'],
    'Ciudad':   ['Nueva York','Nueva York','Chicago','Chicago'],
    'Ventas':   [100,150,150,50]
}

df = pd.DataFrame(data)

# pivot para crear un cubo simple (suma de ventas por fecha y producto)
cube = pd.pivot_table(df,values='Ventas',index='Fecha',columns='Producto',aggfunc='sum')

print("Cubo simple: ")
print(cube)

# se puede realizar un analisis mas completo agregando mas dimensiones
# por ejemplo, podriamos querer saber las ventas por ciudad y por producto

cube_multi_dimension = pd.pivot_table(df,values='Ventas',index=['Fecha','Ciudad'],columns='Producto',aggfunc='sum')

print("\nCubo multi-dimension: ")
print(cube_multi_dimension)

# podriamos querer "rodar" (roll-up) el cubo para tener las ventas por productos
cube_rollup = pd.pivot_table(df,values='Ventas',columns='Producto',aggfunc='sum')

print("\nCubo rodado (rolled-up): ")
print(cube_rollup)

