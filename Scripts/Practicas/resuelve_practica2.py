import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data_alumnos = pd.read_csv("../../Datasets/datos_alumnos.csv")

sns.barplot(x='Materia', y='Calificacion', hue='Semestre', data=data_alumnos)
plt.show()