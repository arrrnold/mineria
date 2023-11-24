from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("../../Datasets/housing.csv")  # leyendo el ds
data = pd.DataFrame(data)

# transformar variables categoricas a numericas
label_encoder = LabelEncoder()
data['ocean_proximity'] = label_encoder.fit_transform(data['ocean_proximity'])

# llenar valores vacios con un promedio de la respectiva columna
data.fillna(data.mean(), inplace=True)

# correlacion
correlation_matrix = data.iloc[:, 2:].corr()  # excluye las dos primeras columnas

# plot de la matriz con un mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.show()

print(correlation_matrix)

# regresion lineal multiple
X = data[
    ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'latitude',
     'longitude', 'ocean_proximity']]
y = data['median_house_value']

# establecer conjuntos de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = LinearRegression()  # crear modelo
model.fit(X_train, y_train)  # entrenar modelo
y_pred = model.predict(X_test)  # prediccion

mse = mean_squared_error(y_test, y_pred)  # evaluacion por ms3
r2 = r2_score(y_test, y_pred)  # evaluacion por r^2

print("Resultados de la predicción:")
print(f"Error cuadrático medio (MSE): {mse:.2f}")
print(f"Coeficiente de determinación (R²): {r2:.2f}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, c='g', label='Precios predichos')
plt.scatter(y_test, y_test, alpha=0.5, c='r', label='Precios reales')
plt.title("Precios reales vs. predichos")
plt.xlabel("Precios reales")
plt.ylabel("Precios predichos")
plt.legend()
plt.show()


# seleccionar las variables a redimensionar
variables = data[
    ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'latitude',
     'longitude', 'ocean_proximity']]

# normalizar las variables utilizando StandardScaler
scaler = StandardScaler()
variables_normalizadas = scaler.fit_transform(variables)

# hacer el ACP en los datos normalizados
acp = PCA()
componentes = acp.fit_transform(variables_normalizadas)

# calcular la varianza explicada por cada variable
varianza_explicada = acp.explained_variance_ratio_

# plot de la varianza explicada por cada variable
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(varianza_explicada) + 1), varianza_explicada, alpha=0.5, align='center')
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada')
plt.title('Varianza Explicada por Componente Principal')
plt.show()

# determinar muestras de cada variable necesarias para conservar cierta cantidad de varianza (por ejemplo, el 95%)
varianza_acumulada = np.cumsum(varianza_explicada)
n_componentes_a_retener = np.argmax(varianza_acumulada >= 0.95) + 1

# reducir la dimensionalidad
datos_reducidos = componentes[:, :n_componentes_a_retener]

# plot de la reducción de dimensionalidad
plt.figure(figsize=(10, 6))
plt.scatter(datos_reducidos[:, 0], datos_reducidos[:, 1], alpha=0.5)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Reducción de Dimensionalidad con ACP')
plt.show()

# regresion lineal multiple con datos reducidos

X = datos_reducidos
y = data['median_house_value']  # variable a predecir

# establecer conjuntos de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_acp = LinearRegression()  # crear modelo
model_acp.fit(X_train, y_train)  # entrenar modelo
y_pred_acp = model_acp.predict(X_test)  # prediccion

mse_acp = mean_squared_error(y_test, y_pred_acp)  # MSE
r2_acp = r2_score(y_test, y_pred_acp)  # R^2

print("Resultados del modelo reducido:")
print(f"Error cuadrático medio (MSE): {mse_acp:.2f}")
print(f"Coeficiente de determinación (R²): {r2_acp:.2f}")

# graficar valores reales y predichos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_acp, alpha=0.5, c='b', label='Valores Predichos')
plt.scatter(y_test, y_test, alpha=0.5, c='r', label='Valores Reales')
plt.title("Valores Reales vs. Predichos")
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.legend()
plt.show()
