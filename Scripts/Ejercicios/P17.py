import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Generamos datos aleatorios
np.random.seed(0)
data = np.random.rand(1000, 2)
labels = (data[:, 0] + data[:, 1]) > 1  # Etiqueta es 1 si la suma de las coordenadas es mayor que 1, 0 en caso contrario

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  # Capa oculta con 4 neuronas y activación ReLU
model.add(Dense(1, activation='sigmoid'))  # Capa de salida con activación sigmoide

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(data, labels, epochs=20, verbose=1)

loss, accuracy = model.evaluate(data, labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

plt.plot(history.history['accuracy'])
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
