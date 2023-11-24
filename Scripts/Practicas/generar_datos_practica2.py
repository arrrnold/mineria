import pandas as pd
import random
from datetime import datetime

# Crear listas con valores posibles para cada columna
materias = ['matematicas','biologia','estadistica','mineria','finanzas']
calificaciones = ['diez', 'nueve', 'ocho', 'siete', 'seis']
semestres = ['primero', 'segundo', 'tercero', 'cuarto', 'quinto']

# Inicializar listas vacías para almacenar los datos generados
lista_materias = []
lista_calificaciones = []
lista_semestres = []

# Generar 500 filas de datos
for _ in range(500):
    materia = random.choice(materias)
    semestre = random.choice(semestres)
    calificacion = random.choice(calificaciones)

    # Añadir los datos generados a las listas
    lista_materias.append(materia)
    lista_semestres.append(semestre)
    lista_calificaciones.append(calificacion)

# Crear un DataFrame de pandas con los datos
df = pd.DataFrame({
    'Materia': lista_materias,
    'Semestre': lista_semestres,
    'Calificacion': lista_calificaciones
})

# Ver las primeras filas del DataFrame para asegurarnos de que se ha creado correctamente
print(df.head())

# Opcional: guardar el DataFrame en un archivo CSV
df.to_csv('datos_alumnos.csv', index=False)


# # Crear un DataFrame simulado df
# df_data = {
#     'Fecha': ['2021-01', '2021-01', '2021-02', '2021-02', '2021-03'],
#     'Producto': ['Manzana', 'Banana', 'Manzana', 'Banana', 'Manzana'],
#     'Region': ['Norte', 'Norte', 'Sur', 'Sur', 'Oeste'],
#     'Ventas': [100, 75, 90, 80, 110]
# }

# df = pd.DataFrame(df_data)

# # Crear un DataFrame simulado df_inventario
# df_inventario_data = {
#     'Fecha': ['2021-01', '2021-01', '2021-02', '2021-02', '2021-03'],
#     'Producto': ['Manzana', 'Banana', 'Manzana', 'Banana', 'Manzana'],
#     'Inventario': [500, 600, 480, 550, 510]
# }

# df_inventario = pd.DataFrame(df_inventario_data)

# # Generar valores aleatorios para el inventario simulado
# for i in range(len(df_inventario)):
#     df_inventario.at[i, 'Inventario'] = random.randint(400, 700)

# # Guardar los DataFrames en archivos CSV
# df.to_csv('df.csv', index=False)
# df_inventario.to_csv('datos_inventario.csv', index=False)

