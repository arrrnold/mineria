# Un cubo de datos
cubo_de_datos = {
    '2023': {
        'Nueva York': {
            'Televisor': 5000,
            'Telefono': 3000,
        },
        'San Francisco': {
            'Televisor': 4000,
            'Telefono':3500,
        },
    },
    '2022': {
        'Televisor': 3900,
        'Telefono': 3400,
    }
}

# Para añadir los datos al cubo
def anadir_datos(tiempo,ubicacion,producto,ventas):
    if tiempo not in cubo_de_datos:
        cubo_de_datos[tiempo] = {}
    if ubicacion not in cubo_de_datos[tiempo]:
        cubo_de_datos[tiempo][ubicacion] = {}
    cubo_de_datos[tiempo][ubicacion][producto] = ventas

# Para consultar el cubo de datos
def consultar_ventas(tiempo,ubicacion,producto):
    try:
        return cubo_de_datos[tiempo][ubicacion][producto]
    except KeyError:
        return "Informacion no disponible"
    
# Permitir al usuario añadir datos
tiempo = input("Ingrese el tiempo (por ejemplo, 2023)")
ubicacion = input("Ingrese la ubicacion (por ejemplo, Nueva York)")
producto = input("Ingrese el tipo de producto (por ejemplo, Televisor)")
ventas = int(input("Ingresa la cantidad total de ventas: "))

anadir_datos(tiempo,ubicacion,producto,ventas)

tiempo_consulta = input("Ingrese el año que desea consultar")
ubicacion_consulta = input("Ingrese la ubicacion que de sea consultar")
producto_consulta = input("Ingrese el producto a consultar")

print("Ventas totales: ",consultar_ventas(tiempo_consulta,ubicacion_consulta,producto_consulta))