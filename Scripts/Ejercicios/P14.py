
"""# 14 maquina de vectores de soporte"""

# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.datasets import load_iris              # Importar el conjunto de datos Iris
from sklearn.model_selection import train_test_split # Importar función para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.ensemble import RandomForestClassifier  # Importar el clasificador Random Forest
from sklearn.metrics import accuracy_score, classification_report # Importar métricas de evaluación
from sklearn.preprocessing import StandardScaler    # Importar la clase para estandarizar datos
from sklearn.svm import SVC                        # Importar el modelo de Máquina de Vectores de Soporte (SVM)

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data     # Características
y = iris.target   # Etiquetas

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)  # El conjunto de prueba tiene un 30% de los datos y se establece una semilla aleatoria para reproducibilidad

# Estandarizar los datos (escalar)
scaler = StandardScaler()       # Crear un objeto StandardScaler
scaler.fit(X_train)             # Calcular la media y la desviación estándar en el conjunto de entrenamiento
X_train = scaler.transform(X_train)  # Aplicar la estandarización a los datos de entrenamiento
X_test = scaler.transform(X_test)    # Aplicar la estandarización a los datos de prueba

# Crear un modelo SVM con kernel lineal y semilla aleatoria
svm_model = SVC(kernel='linear', random_state=42)  # Crear un modelo SVM con kernel lineal y semilla aleatoria

# Entrenar el modelo SVM en los datos de entrenamiento
svm_model.fit(X_train, y_train)  # Ajustar el modelo a los datos de entrenamiento

# Realizar predicciones en el conjunto de prueba
y_pred = svm_model.predict(X_test)  # Realizar predicciones en los datos de prueba
