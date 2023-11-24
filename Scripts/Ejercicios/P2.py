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

def consultar_ventas(tiempo,ubicacion,producto):
    try:
        return cubo_de_datos[tiempo][ubicacion][producto]
    except KeyError:
        return "Informacion no disponible"
    
print("Ventas de televisores en NY en 2023: $",consultar_ventas('2023','Nueva York','Televisor'))