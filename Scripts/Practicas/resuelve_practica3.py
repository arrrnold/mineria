# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.datasets import load_iris  # Importar el conjunto de datos Iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import \
    train_test_split  # Importar función para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.ensemble import RandomForestClassifier  # Importar el clasificador Random Forest
from sklearn.metrics import accuracy_score, classification_report  # Importar métricas de evaluación
from sklearn.preprocessing import StandardScaler  # Importar la clase para estandarizar datos
from sklearn.svm import SVC  # Importar el modelo de Máquina de Vectores de Soporte (SVM)
from sklearn.tree import DecisionTreeClassifier

# Cargue el conjunto de datos Iris.
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas

# Divida el conjunto de datos en un conjunto de entrenamiento (80% de los datos) y un conjunto de prueba (20% de los
# datos).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implemente y entrene un modelo de Regresión Logística y un modelo de Árbol de Decisiones usando el conjunto de
# entrenamiento.

# Estandarizar los datos (escalar)
scaler = StandardScaler()  # Crear un objeto StandardScaler
scaler.fit(X_train)  # Calcular la media y la desviación estándar en el conjunto de entrenamiento
X_train = scaler.transform(X_train)  # Aplicar la estandarización a los datos de entrenamiento
X_test = scaler.transform(X_test)  # Aplicar la estandarización a los datos de prueba

# Crear los clasificadores
log_reg = LogisticRegression(random_state=42)
clf = DecisionTreeClassifier(random_state=42)

# Entrenar los clasificadores
log_reg.fit(X_train, y_train)
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_reg = log_reg.predict(X_test)
y_pred_arboles = clf.predict(X_test)

# Calcular las precisiones
accuracy_reg_logistica = accuracy_score(y_test, y_pred_reg)
accuracy_arboles = accuracy_score(y_test, y_pred_arboles)

# Imprimir la precisión en porcentaje
print(f'Precisión de la regresión logística usando el DS original: {accuracy_reg_logistica * 100:.2f}%')
print(f'Precisión de los arboles usando el DS original: {accuracy_arboles * 100:.2f}%')

# Implemente una función que acepte un modelo y una nueva observación (características de
# una flor), y retorne el tipo de pétalo predicho por el modelo.
def funcion(modelo, observacion):
    # predecir el tipo de petalo de la nueva observacion
    X_test_funcion = observacion

    # declarar datos de prueba y entrenamiento
    X_train, X_test_funcion, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar los datos (escalar)
    scaler = StandardScaler()  # Crear un objeto StandardScaler
    scaler.fit(X_train)  # Calcular la media y la desviación estándar en el conjunto de entrenamiento
    X_train = scaler.transform(X_train)  # Aplicar la estandarización a los datos de entrenamiento
    X_test_funcion = scaler.transform(X_test_funcion)  # Aplicar la estandarización a los datos de prueba

    # Entrenar los clasificadores
    modelo.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    pred_regresion = modelo.predict(X_test_funcion)

    return pred_regresion

# Permita la entrada de nuevas observaciones y utilice ambos modelos para predecir el tipo
# de pétalo.

long_sepalo = (input("Ingrese la longitud del sépalo (cm): "))
ancho_sepalo = (input("Ingrese el ancho del sépalo (cm): "))
long_petalo = (input("Ingrese la longitud del pétalo (cm): "))
ancho_petalo = (input("Ingrese el ancho del pétalo (cm): "))

nueva_observacion = [long_sepalo, ancho_sepalo, long_petalo, ancho_petalo]

nuevo_modelo_regresion = LogisticRegression(random_state=1)
nuevo_modelo_arboles = DecisionTreeClassifier(random_state=1)

prediccion_regresion_funcion = funcion(nuevo_modelo_regresion, nueva_observacion)
prediccion_arboles_funcion = funcion(nuevo_modelo_arboles, nueva_observacion)

accuracy_reg_logistica_funcion = accuracy_score(y_test, prediccion_regresion_funcion)
accuracy_arboles_funcion = accuracy_score(y_test, prediccion_arboles_funcion)

print(prediccion_regresion_funcion)
print(prediccion_arboles_funcion)

print(f'Precision de los arboles con la nueva observación: {accuracy_reg_logistica * 100:.2f}%')
print(f'Precision de la regresion con la nueva observación: {accuracy_arboles * 100:.2f}%')