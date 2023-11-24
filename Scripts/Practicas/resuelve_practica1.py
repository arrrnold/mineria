import pandas as pd

df=pd.read_csv('../../Datasets/datos_ventas.csv')
df2=pd.read_csv('../../Datasets/datos_inventario.csv')

# 1. Drill-Down
dd = df.groupby(['Fecha', 'Categoría', 'Producto'])['Ventas'].sum().reset_index()
print("1. Drill-Down Result:")
print(dd)

# 2. Roll-Up
ventas_region = df.groupby('Municipio')['Ventas'].sum().reset_index()
ventas_totales = ventas_region['Ventas'].sum()
print("\n2. Roll-Up Result:")
print(ventas_region)
print(f"Ventas Totales: {ventas_totales}")

# 3. Slice and Dice
slice_result = df[(df['Fecha'].str.startswith('2021-01')) & (df['Municipio'] == 'Guadalupe')]
dice_result = slice_result[slice_result['Categoría'] == 'Frutas']
print("\n3. Slice and Dice Result:")
print(dice_result)

# 4. Pivot (o Rotate)
pivot_result = pd.pivot_table(df, values='Ventas', index=['Vendedor'], columns=['Fecha'], aggfunc=sum, fill_value=0)
print("\n4. Pivot Result:")
print(pivot_result)


# 5. Drill-Through
total_ventas_frutas_2021_01 = df[(df['Fecha'].str.startswith('2021-01')) & (df['Categoría'] == 'Frutas')]['Ventas'].sum()
drill_through_result = df[(df['Fecha'].str.startswith('2021-01')) & (df['Categoría'] == 'Frutas')]
print("\n5. Drill-Through Result:")
print(drill_through_result)

# 6. Drill-Across (Simulación ya que no tenemos un DataFrame real de inventario)
# El df_inventario es el otro conjunto de datos.
print("\n6. Drill-Across Result:")
drill_across_result = pd.merge(df, df2, on=['Producto', 'Fecha'])
print(drill_across_result)

correlation = drill_across_result['Ventas'].corr(drill_across_result['Inventario'])
print(f"La correlación entre Ventas e Inventario es: {correlation:.2f}")

# 7. Consolidación
consolidation_result = df.groupby('Vendedor').agg({'Ventas': ['sum', 'mean', 'max'], 'UnidadesVendidas': ['sum', 'mean', 'max']}).reset_index()
print("\n7. Consolidación Result:")
print(consolidation_result)



import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x='Materia', y='Calificación', hue='Semestre', data=cubo)
plt.show()


calif_materia = df.groupby(['Semestre', 'Materia'])['Calificacion'].mean().reset_index()
print(calif_materia)