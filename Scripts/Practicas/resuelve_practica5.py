# Objetivo: Predecir el `Type 1` de un Pokémon basándonos en sus estadísticas (HP,
# Attack, Defense, SP Atk, SP Def, Speed).
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# COMPRENSION DE LOS DATOS

# Exploración del dataset
pokemon = pd.read_csv('../../Datasets/pokemon.csv')  # leer ds
pokemon = pd.DataFrame(pokemon)  # hacerlo df

columna_Type1 = pokemon['Type 1']
tipos_de_pokemon = columna_Type1.value_counts()
total_de_pokemones = pokemon['Type 1'].count()
print("Tipos de pokemones\n", tipos_de_pokemon)
print("Total de pokemones: ", total_de_pokemones)

# LIMPIEZA DE LOS DATOS
pokemon = pokemon.fillna('NA')  # llena los valores faltantes con "NA"

# Asignar un valor numerico a cada dato de type 1
mapeo_tipo = {}
contador = 1

# Iterar a través de los tipos de Pokémon únicos y asignarles valores numéricos
for tipo in tipos_de_pokemon.index:
    mapeo_tipo[tipo] = contador
    contador += 1

# Aplicar el mapeo a la columna 'Type 1'
pokemon['Type 1'] = pokemon['Type 1'].map(mapeo_tipo)

# Imprimir el mapeo
print("Mapeo de tipos de Pokémon a números:")
for tipo, numero in mapeo_tipo.items():
    print(f"{tipo} -> {numero}")

# Para convertir valores categoricos en numericos
label_encoder = LabelEncoder()

# Codificar las columnas categóricas (ajusta los nombres de las columnas según tu dataset)
pokemon['Name'] = label_encoder.fit_transform(pokemon['Name'])
pokemon['Type 2'] = label_encoder.fit_transform(pokemon['Type 2'])
pokemon['Legendary'] = label_encoder.fit_transform(pokemon['Legendary'])

# MODELADO

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = pokemon.drop(columns=['Type 1'])
y = pokemon['Type 1']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear clasificador
clf = RandomForestClassifier(n_estimators=300, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Graficar la predicción
plt.figure(figsize=(12, 6))

# Obtener los IDs de los Pokémon en el conjunto de prueba
pokemon_ids = X_test.index

# Graficar las etiquetas reales
plt.scatter(pokemon_ids, y_test, c='blue', marker='o', label='Tipo Real')

# Graficar las etiquetas predichas
plt.scatter(pokemon_ids, y_pred, c='red', marker='x', s=100, label='Tipo Predicho')

# Configuración de la gráfica
plt.xlabel('ID del Pokémon')
plt.ylabel('Tipo de Pokémon')
plt.title('Predicciones del modelo de bosques aleatorios')
plt.legend(['Tipo Real', 'Tipo Predicho'])
plt.grid(True)

# Mostrar la gráfica
plt.show()

# EVALUACIÓN DEL MODELO
accuracy = accuracy_score(y_test, y_pred) # calcular precisión del modelo
print("Porcentaje de precisión del modelo:", accuracy * 100) # mostrar % de precision

# Crear la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)

# Crear un mapa de calor para visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=range(1, 19), yticklabels=range(1, 19))
plt.xlabel('Etiquetas predichas')
plt.ylabel('Etiquetas reales')
plt.title('Matriz de confusión')
plt.show()