import pandas as pd

data = {
    'Fecha': ['2021-01', '2021-01', '2021-02', '2021-02', '2021-03'],
    'Producto': ['Manzana', 'Banana', 'Manzana', 'Banana', 'Manzana'],
    'Region': ['Norte', 'Norte', 'Sur', 'Sur', 'Oeste'],
    'Ventas': [100, 75, 90, 80, 110]
}

df = pd.DataFrame(data)
print("DataFrame Original:")
print(df)

#Drill-Down: Detallar las ventas por mes y producto
print("\nDrill-Down (Ventas por Mes y Producto):")
drill_down = df.groupby(['Fecha', 'Producto']).sum()
print(drill_down)

#Roll-Up: Resumir las ventas por mes
print("\nRoll-Up (Ventas por Mes):")
roll_up = df.groupby('Fecha').sum()
print(roll_up)

#Slice: Seleccionar ventas en el mes '2021-01'
print("\nSlice (Ventas en Enero 2021):")
slice_op = df[df['Fecha'] == '2021-01']
print(slice_op)

#Dice: Seleccionar ventas de 'Manzana' en el mes '2021-01'
print("\nDice (Ventas de Manzanas en Enero 2021):")
dice_op = df[(df['Producto'] == 'Manzana') & (df['Fecha'] == '2021-01')]
print(dice_op)

#Pivote: Cambiar las dimensiones de Fecha y Producto
print("\nPivot (Cambiar dimensiones):")
pivot_op = pd.pivot_table(df, values='Ventas', index='Producto', 
                          columns='Fecha', aggfunc='sum')
print(pivot_op)

#Drill-Through: Mostrar los datos detallados que componen una suma de ventas
print("\nDrill-Through (Datos que componen la suma de ventas en Enero 2021):")
drill_through = df[df['Fecha'] == '2021-01']
print(drill_through)

#Drill-Across: Esto requiere dos DataFrames, 
# vamos a crear un segundo DataFrame para el ejemplo
data2 = {
    'Fecha': ['2021-01', '2021-01', '2021-02', '2021-02', '2021-03'],
    'Producto': ['Manzana', 'Banana', 'Manzana', 'Banana', 'Manzana'],
    'Inventario': [200, 150, 180, 160, 220]
}
df2 = pd.DataFrame(data2)
#Aquí cruzamos datos de Ventas e Inventario para el producto 'Manzana'
print("\nDrill-Across (Cruzar datos de Ventas e Inventario para 'Manzana'):")
drill_across = pd.merge(
    df[df['Producto'] == 'Manzana'],
    df2[df2['Producto'] == 'Manzana'],
    on=['Fecha', 'Producto']
)
print(drill_across)

#Consolidación: Calcular métricas como la suma total, el promedio, etc.
print("\nConsolidación (Suma y Promedio de Ventas):")
consolidation = df['Ventas'].agg(['sum', 'mean'])
print(consolidation)
