"""# 12 Regresion logistica

Este código realiza lo siguiente:

Carga el conjunto de datos Iris y crea etiquetas binarias donde 1 representa la especie Virginica y 0 representa las otras especies.

Divide los datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split.

Estandariza (escala) las características utilizando StandardScaler para asegurarse de que tengan una distribución similar.

Crea un clasificador de regresión logística utilizando LogisticRegression de scikit-learn.

Entrena el clasificador en los datos de entrenamiento utilizando fit.

Realiza predicciones en el conjunto de prueba utilizando predict.

Calcula la precisión del modelo utilizando accuracy_score.

Imprime la precisión en porcentaje como resultado de la evaluación del modelo de regresión logística en el conjunto de prueba.**texto en negrita**
"""

# Importar las bibliotecas necesarias
import numpy as np                      # Importar numpy para manipulación de arreglos
from sklearn import datasets           # Importar datasets de scikit-learn
from sklearn.model_selection import train_test_split  # Importar función para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler     # Importar la clase para estandarizar datos
from sklearn.linear_model import LogisticRegression  # Importar el clasificador de regresión logística
from sklearn.metrics import accuracy_score          # Importar métricas de evaluación

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()

# Obtener las características (X) y las etiquetas (y) del conjunto de datos
X = iris.data
y = (iris.target == 2).astype(int)  # Crear etiquetas binarias: 1 si Virginica, 0 si no

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)  # El conjunto de prueba tiene un 20% de los datos y se establece una semilla aleatoria para reproducibilidad

# Estandarizar los datos (escalar)
scaler = StandardScaler()       # Crear un objeto StandardScaler
scaler.fit(X_train)             # Calcular la media y la desviación estándar en el conjunto de entrenamiento
X_train = scaler.transform(X_train)  # Aplicar la estandarización a los datos de entrenamiento
X_test = scaler.transform(X_test)    # Aplicar la estandarización a los datos de prueba

# Crear un clasificador de regresión logística
log_reg = LogisticRegression(random_state=42)

# Entrenar el clasificador en los datos de entrenamiento
log_reg.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = log_reg.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Imprimir la precisión en porcentaje
print(f'Accuracy: {accuracy * 100:.2f}%')