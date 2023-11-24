# inicializar un diccionario vacio para actuar como nuestro cubo de datos
cubo_de_datos = {}

# Funcion para añadir datos al cubo
def anadir_datos(tiempo,ubicacion,producto,ventas):
    if tiempo not in cubo_de_datos:
        cubo_de_datos[tiempo] = {}
    if ubicacion not in cubo_de_datos[tiempo]:
        cubo_de_datos[tiempo][ubicacion] = {}
    cubo_de_datos[tiempo][ubicacion][producto] = ventas

# funcion para mostrar el cubo de datos
def mostrar_cubo():
    for tiempo, datos_tiempo in cubo_de_datos.items():
        print(f"Año: {tiempo}")
        for ubicacion, datos_ubicacion in datos_tiempo.items():
            print(f"Ubicacion: {ubicacion}")
            for producto, ventas in datos_ubicacion.items():
                print(f" Producto: {producto}, Ventas Totales: {ventas}")

# interaccion con el usuario para llenar el cubo de datos
while True:
    print("---- Llenar cubo de datos ----")

    tiempo = input("Ingrese el tiempo (por ejemplo, 2023)")
    ubicacion = input("Ingrese la ubicacion (por ejemplo, Nueva York)")
    producto = input("Ingrese el tipo de producto (por ejemplo, Televisor)")
    ventas = int(input("Ingresa la cantidad total de ventas: "))

    anadir_datos(tiempo, ubicacion, producto, ventas)

    mostrar_cubo()

    continuar = input("¿Desea continuar añadiendo datos? (s/n): ")
    if continuar.lower() != 's':
        break