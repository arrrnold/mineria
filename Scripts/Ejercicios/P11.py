"""# 11 No hagas iris
Este codigo:
1. Carga el conjunto de datos Iris, que es un conjunto de datos de clasificación con características de flores y etiquetas de especies.

2. Divide los datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split.

3. Crea un clasificador de árbol de decisión utilizando DecisionTreeClassifier de scikit-learn.

4. Entrena el clasificador en los datos de entrenamiento utilizando fit.

5. Realiza predicciones en el conjunto de prueba utilizando predict.

6. Calcula la precisión del modelo utilizando accuracy_score.

7. Calcula la matriz de confusión del modelo utilizando confusion_matrix.

8. Imprime la precisión en porcentaje y la matriz de confusión como resultados de la evaluación del modelo de árbol de decisión en el conjunto de prueba.
"""

# Importar las bibliotecas necesarias
from sklearn.datasets import load_iris  # Importar el conjunto de datos Iris
from sklearn.model_selection import train_test_split  # Importar función para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.tree import DecisionTreeClassifier  # Importar el clasificador de árbol de decisión
from sklearn.metrics import accuracy_score, confusion_matrix  # Importar métricas de evaluación

# Cargar el conjunto de datos Iris
iris = load_iris()

# Obtener las características (X) y las etiquetas (y) del conjunto de datos
X, y = iris.data, iris.target

# Dividir los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)  # El conjunto de prueba tiene un 30% de los datos y se establece una semilla aleatoria para reproducibilidad

# Crear un clasificador de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Entrenar el clasificador en los datos de entrenamiento
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Imprimir los resultados
print(f'Accuracy: {accuracy * 100:.2f}%')  # Precisión en porcentaje
print(f'Confusion Matrix:\n{conf_matrix}')  # Matriz de confusión