
"""# 16 Una red neuronal

El modelo de red neuronal se entrena durante 50 épocas en los datos de entrenamiento. Cada época representa una pasada completa a través de los datos de entrenamiento durante el proceso de entrenamiento.

Durante el entrenamiento, se observa la pérdida (loss) y la precisión (accuracy) en cada época. Estos valores se imprimen en la salida.

Después de las 50 épocas, el modelo se evalúa en los datos de prueba utilizando model.evaluate(). Esto se hace para evaluar el rendimiento del modelo en un conjunto de datos que no ha visto durante el entrenamiento.

La salida muestra que la pérdida en los datos de prueba es aproximadamente 0.481 y la precisión es aproximadamente 0.778 (77.8%).
"""

# Importar las bibliotecas necesarias
import tensorflow as tf  # Importar TensorFlow, una biblioteca de aprendizaje automático y deep learning
from sklearn.datasets import load_iris  # Importar el conjunto de datos Iris
from sklearn.model_selection import train_test_split  # Importar función para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler  # Importar la clase para estandarizar datos
from sklearn.preprocessing import OneHotEncoder  # Importar la clase para estandarizar datos

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas

# Codificar las etiquetas utilizando One-Hot Encoding
encoder = OneHotEncoder(sparse=False)  # Crear un objeto OneHotEncoder
y = encoder.fit_transform(y.reshape(-1, 1))  # Aplicar One-Hot Encoding a las etiquetas y transformarlas

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)  # El conjunto de prueba tiene un 30% de los datos y se establece una semilla aleatoria para reproducibilidad

# Estandarizar los datos (escalar)
scaler = StandardScaler()       # Crear un objeto StandardScaler
scaler.fit(X_train)             # Calcular la media y la desviación estándar en el conjunto de entrenamiento
X_train = scaler.transform(X_train)  # Aplicar la estandarización a los datos de entrenamiento
X_test = scaler.transform(X_test)    # Aplicar la estandarización a los datos de prueba

# Construir el modelo de red neuronal
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=8, activation='relu', input_shape=(4,)),  # Capa oculta con 8 neuronas y función de activación ReLU
    tf.keras.layers.Dense(units=3, activation='softmax')  # Capa de salida con 3 neuronas y función de activación Softmax
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50)  # Entrenar el modelo en los datos de entrenamiento durante 50 épocas

# Evaluar el modelo en los datos de prueba
loss, accuracy = model.evaluate(X_test, y_test)  # Evaluar el modelo en los datos de prueba y obtener la pérdida y la precisión
print(f"Loss: {loss}")  # Imprimir la pérdida
print(f"Accuracy: {accuracy}")  # Imprimir la precisión
