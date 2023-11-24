import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/online_shoppers_intention.csv")
data = pd.read_csv("../../Datasets/online_shoppers_intention.csv")

# variables categoricas a numericas
encoder = LabelEncoder()
data['Month'] = encoder.fit_transform(data['Month'])
data['VisitorType'] = encoder.fit_transform(data['VisitorType'])
data['Weekend'] = encoder.fit_transform(data['Weekend'])

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # normalizar datos

# separar datos
X = data_scaled[:, :17]  # tomar las otras 17 variables para predecir la 18
y = data['Revenue']  # 'Revenue' es la columna que representa las etiquetas en este caso
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

####### MODELO DE REGRESION LOGISTICA #######

# inicializar
# W y b se inicializan con valores aleatorios
W = np.random.rand(X.shape[1])
b = np.random.rand(1)[0]

taza = 0.01  # taza de aprendizaje
max_iter = 1000  # numero de iteraciones
critero = 0.0001  # criterio de convergencia


# definir la funcion sigmoidea
def sigmoidea(z):
    return 1 / (1 + np.exp(-z))


for i in range(max_iter):
    # calcular el modelo
    z = np.dot(X_train, W) + b
    y_pred = sigmoidea(z)

    # calcular el error
    error = y_pred - y_train

    # calcular el gradiente
    gradiente_W = np.dot(X_train.T, error) / len(X_train)
    gradiente_b = np.sum(error) / len(X_train)

    # actualizar los pesos
    W = W - taza * gradiente_W  # pesos de las variables
    b = b - taza * gradiente_b  # bias aka sesgo

    # calcular el error cuadratico medio
    error_cuad_medio = np.mean(np.power(error, 2))

    # detener el entrenamiento si el error es menor al criterio
    if error_cuad_medio < critero:
        break


# funcion de prediccion
def predecir(X):
    z = np.dot(X, W) + b
    y_pred = sigmoidea(z)

    # clasificar las salidas
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    return y_pred


y_pred = predecir(X_train)  # entrenar modelo con datos de entrenamiento
y_pred_test = predecir(X_test)  # evaluar modelo con datos de prueba

plt.figure(figsize=(10, 6))

plt.scatter(range(len(y_train)), y_train, color='blue', label='Etiqueta Real')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicción')

plt.title('Predicciones del Modelo vs Etiquetas Reales', fontsize=16)
plt.xlabel('Instancias', fontsize=14)
plt.ylabel('Clase', fontsize=14)
plt.legend()
plt.show()

# classification report
print(classification_report(y_train, y_pred))

# matriz de confusion
matriz_confusion = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 8))
sns.heatmap(matriz_confusion, annot=True, cbar=False,
            fmt='d', cmap='Blues', annot_kws={'size': 20},
            square=True, linewidths=0.5,
            xticklabels=['No compra`', 'Si compra'],
            yticklabels=['No compra`', 'Si compra'])

plt.title('Matriz de Confusión', fontsize=20)
plt.xlabel('Predicción', fontsize=20)
plt.ylabel('Etiqueta', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
