# NORMALIZACION DE DATOS CON Z-SCORE
import pandas as pd
from sklearn.preprocessing import StandardScaler  # el z score

data = {
    'edad': [25, 30, 35, 40, 45],
    'salario': [50000, 55000, 58000, 62000, 64000]
}

df = pd.DataFrame(data)
scaler = StandardScaler()
df_standarized = scaler.fit_transform(df)
df_standarized = pd.DataFrame(df_standarized, columns=df.columns)

print(data)
print(df_standarized)
