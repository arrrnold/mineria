"""# 10 Una Regresion lineal"""

# Importar las bibliotecas necesarias
import numpy as np                  # Importar numpy para manipulación de arreglos
import matplotlib.pyplot as plt     # Importar matplotlib para visualización
from sklearn import datasets        # Importar datasets de scikit-learn
from sklearn.metrics import mean_squared_error, r2_score  # Importar métricas para evaluar el modelo
from sklearn.model_selection import train_test_split  # Importar función para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn import datasets, linear_model


# Cargar el conjunto de datos Iris
iris = datasets.load_iris()

# Seleccionar las características de interés (longitud del sépalo)
x = iris.data[:, np.newaxis, 0]

# Seleccionar la variable de respuesta (ancho del sépalo)
y = iris.data[:, np.newaxis, 1]

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=20, random_state=42)

# Crear un modelo de regresión lineal
regresion_lineal = linear_model.LinearRegression()

# Entrenar el modelo en los datos de entrenamiento
regresion_lineal.fit(x_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = regresion_lineal.predict(x_test)

# Imprimir los coeficientes del modelo
print('Coefficients: \n', regresion_lineal.coef_)

# Calcular y mostrar el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error: %.2f' % mse)

# Calcular y mostrar el coeficiente de determinación (R^2)
r2 = r2_score(y_test, y_pred)
print('Coeficiente de determinación (R^2): %.2f' % r2)

# Visualización de los datos y la línea de regresión
plt.scatter(x_test, y_test, color='black')  # Puntos de prueba en negro
plt.plot(x_test, y_pred, color='blue', linewidth=3)  # Línea de regresión en azul
plt.xticks(())
plt.yticks(())

plt.show()
