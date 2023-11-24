import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
data = pd.read_csv("../../Datasets/healthcare_dataset.csv")

# Observar las primeras filas
print(data.head())

# Estadísticas descriptivas
print(data.info())
print(data.describe())

# Visualización de datos
# Por ejemplo, histograma de la columna 'Age'
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Distribución de Edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()
