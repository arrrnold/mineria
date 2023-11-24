# NORMALIZACION DE DATOS CON MAXIMOS Y MINIMOS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    'edad': [25, 30, 35, 40, 45],
    'salario': [50000, 55000, 58000, 62000, 64000]
}

df = pd.DataFrame(data)
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

print(data)
print(df_scaled)
