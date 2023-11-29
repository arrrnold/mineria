import pandas as pd

#
# determinaciones_bioquimicas = pd.read_csv("Determinaciones_bioquímicas_cronicas_deficiencias_9feb23.csv", sep=";")
# print(determinaciones_bioquimicas.head())
#
# actividad_fisica = pd.read_csv("ensafisica2022_adultos_entrega_w.csv", sep=";")
# print(actividad_fisica.head())
#
# tension_arterial = pd.read_csv("ensaantro2022_entrega_w.csv", sep=";")
# print(tension_arterial.head())
#
# # encontrar los folios de determinaciones bioquimicas que coinciden con los de actividad fisica y tension arterial
# # 1. encontrar los folios
# # 2. hacer un nuevo dataset con los que si coinciden
#
# # Encuentra los folios comunes entre ambos conjuntos de datos
# folios_comunes = set(determinaciones_bioquimicas['FOLIO_I']).intersection(actividad_fisica['FOLIO_I']).intersection(
#     tension_arterial['FOLIO_I'])
#
# # Filtra las muestras que tienen folios en común
# dataset_fusionado = pd.merge(determinaciones_bioquimicas[determinaciones_bioquimicas['FOLIO_I'].isin(folios_comunes)],
#                              actividad_fisica[actividad_fisica['FOLIO_I'].isin(folios_comunes)],
#                              on='FOLIO_I', how='inner')
# dataset_fusionado = pd.merge(dataset_fusionado, tension_arterial[tension_arterial['FOLIO_I'].isin(folios_comunes)],
#                              on='FOLIO_I', how='inner')
#
# dataset_fusionado.to_csv("dataset_fusionado.csv", index=False)  # Guarda el dataset fusionado

data = pd.read_csv("dataset_fusionado.csv", low_memory=False)  # Lee el dataset fusionado
print(data.head())

# 1. ¿Cuántas muestras hay en el dataset?
print("Hay {} muestras en el dataset".format(len(data)))

# 2. ¿Cuántas variables hay en el dataset?
print("Hay {} variables en el dataset".format(len(data.columns)))

# limpieza de datos
print(("########## Limpieza de datos ##########"))
# 1. ¿Cuántos valores faltantes hay por variable?
print(data.isnull().sum())

# eliminar las variables que tienen más de 1000 valores faltantes
data = data.dropna(axis=1, thresh=1000)
print(data.isnull().sum())