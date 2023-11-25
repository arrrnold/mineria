import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


############ RED NEURONAL ADALINE ############

def activacion(z):
    return z


def costo(y, y_pred):
    return np.sum((y - y_pred) ** 2)


def entrenar(X, y, W, tasa_aprendizaje=0.01, epocas=100):
    X_bias = np.c_[np.ones(X.shape[0]), X]  # agrega el sesgo al conjunto de datos

    for epoca in range(epocas):
        for i in range(X_bias.shape[0]):
            z = np.dot(X_bias[i], W)
            y_pred = activacion(z)
            W += tasa_aprendizaje * (y[i] - y_pred) * X_bias[i]  # actualizar los W

    return W


def predecir(X_nuevo, W, umbral=0.5):
    X_bias = np.c_[np.ones(X_nuevo.shape[0]), X_nuevo]  # agrega el sesgo al conjunto de datos
    entrada_neta = np.dot(X_bias, W)
    y_pred = activacion(entrada_neta)
    clases = np.where(y_pred >= umbral, 1, 0)  # clasificación binaria
    return clases


############ USAR LA RED ############

# cargar datos
# data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/online_shoppers_intention.csv")
data = pd.read_csv("../../Datasets/online_shoppers_intention.csv")

# variables categóricas a numéricas
codificador = LabelEncoder()
data['Month'] = codificador.fit_transform(data['Month'])
data['VisitorType'] = codificador.fit_transform(data['VisitorType'])
data['Weekend'] = codificador.fit_transform(data['Weekend'])

escalador = StandardScaler()
datos_normalizados = escalador.fit_transform(data)

# separar datos
X = datos_normalizados[:, :17]  # tomar las otras 17 variables para predecir la 18
y = data['Revenue']  # variable a predecir

# Convertir y a un array de numpy para evitar problemas de indexación
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # dividir datos

# inicializar pesos (W)
tamano_entrada = X_train.shape[1]
W = np.random.rand(tamano_entrada + 1)

W = entrenar(X_train, y_train, W)
predicciones = predecir(X_test, W)

# evaluar modelo
informe = classification_report(y_test, predicciones)
print(informe)

# Graficar predicciones versus datos reales
plt.scatter(range(len(y_test)), y_test, label='Datos Reales', marker='o', color='blue')
plt.scatter(range(len(predicciones)), predicciones, label='Predicciones', marker='x', color='red')
plt.xlabel('Muestras')
plt.ylabel('Clase')
plt.title('Predicciones vs Datos Reales')
plt.legend()
plt.show()

# graficar matriz de confusion
matriz_confusion = confusion_matrix(y_test, predicciones)
sns.heatmap(matriz_confusion, annot=True,
            cmap='Greens', fmt='d', cbar=False,
            xticklabels=['No', 'Si'], yticklabels=['No', 'Si'],
            annot_kws={'size': 20}, square=True, linewidths=0.5)
plt.xlabel('Predicciones')
plt.ylabel('Datos Reales')
plt.title('Matriz de Confusión')
plt.show()

############ MODELO CON APRENDIZAJE AUTOMATICO ############

# usar random forests
modelo = RandomForestClassifier(n_estimators=100, max_depth=10)
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)

# graficar prediccion
plt.scatter(range(len(y_test)), y_test, label='Datos Reales', marker='o', color='blue')
plt.scatter(range(len(predicciones)), predicciones, label='Predicciones', marker='x', color='red')
plt.xlabel('Muestras')
plt.ylabel('Clase')
plt.title('Predicciones vs Datos Reales')
plt.legend()
plt.show()

# evaluar modelo
informe = classification_report(y_test, predicciones)
print(informe)

# graficar matriz de confusion
matriz_confusion = confusion_matrix(y_test, predicciones)
sns.heatmap(matriz_confusion, annot=True,
            cmap='Reds', fmt='d', cbar=False,
            xticklabels=['No', 'Si'], yticklabels=['No', 'Si'],
            annot_kws={'size': 20}, square=True, linewidths=0.5)
plt.xlabel('Predicciones')
plt.ylabel('Datos Reales')
plt.title('Matriz de Confusión')
plt.show()
