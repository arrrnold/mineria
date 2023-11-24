"""# 15 KNN"""

# Importar las bibliotecas necesarias
from sklearn.datasets import load_iris  # Importar el conjunto de datos Iris
from sklearn.model_selection import train_test_split  # Importar función para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler  # Importar la clase para estandarizar datos
from sklearn.neighbors import KNeighborsClassifier  # Importar el clasificador KNeighbors
from sklearn.metrics import accuracy_score, classification_report  # Importar métricas de evaluación

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)  # El conjunto de prueba tiene un 30% de los datos y se establece una semilla aleatoria para reproducibilidad

# Estandarizar los datos (escalar)
scaler = StandardScaler()       # Crear un objeto StandardScaler
scaler.fit(X_train)             # Calcular la media y la desviación estándar en el conjunto de entrenamiento
X_train = scaler.transform(X_train)  # Aplicar la estandarización a los datos de entrenamiento
X_test = scaler.transform(X_test)    # Aplicar la estandarización a los datos de prueba

# Crear un modelo KNeighborsClassifier con 5 vecinos cercanos
knn_model = KNeighborsClassifier(n_neighbors=5)  # Crear un modelo KNeighborsClassifier con 5 vecinos cercanos

# Entrenar el modelo KNeighborsClassifier en los datos de entrenamiento
knn_model.fit(X_train, y_train)  # Ajustar el modelo a los datos de entrenamiento

# Realizar predicciones en el conjunto de prueba
y_pred = knn_model.predict(X_test)  # Realizar predicciones en los datos de prueba

# Calcular y mostrar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Imprimir el informe de clasificación
print("Informe de Clasificación:")
print(classification_report(y_test, y_pred))

# Ahora puedes agregar la matriz de confusión si lo deseas
# Por ejemplo, utilizando la función confusion_matrix de Scikit-Learn
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(confusion)
