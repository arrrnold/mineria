import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris  # Importar el conjunto de datos Iris
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier  # Importar el clasificador Random Forest
from sklearn.metrics import accuracy_score, classification_report  # Importar métricas de evaluación
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Importar la clase para estandarizar datos
from sklearn.svm import SVC  # Importar el modelo de Máquina de Vectores de Soporte (SVM)
from sklearn.tree import DecisionTreeClassifier

# generacion de los datos
np.random.seed(42)
n_samples = 1000
edades = np.random.randint(18, 70, n_samples)
salarios = np.random.randint(20000, 100000, n_samples)
tipo_producto = np.random.choice(['A', 'B', 'C'], n_samples)
interacciones = np.random.randint(1, 20, n_samples)

# empezar a transformar los datos
churn = np.where(
    (salarios < 30000) |
    (interacciones > 15), 1, 0
)

# generar el df
df = pd.DataFrame({
    'edad': edades,
    'salario': salarios,
    'tipo_producto': tipo_producto,
    'interacciones': interacciones,
    'Churn': churn
})

indices_faltantes = np.random.choice(df.index, size=int(0.05 * n_samples),
                    replace=False) # llenar la bd con N/A a los campos q estan vacios

# preprocesamiento

df['salario'].fillna(df['salario'].mean(), inplace=True)
df = df[df['edad'] < 120]

# transformacion

# para interpretar las variables cat. y numericas
cat_features = ['tipo_producto']
num_features = ['edad', 'salario', 'interacciones']

# crear transformadores para variables categoricas y numericas
transformers = [
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first'), cat_features)
]

preprocessor = ColumnTransformer(transformers)

# mineria de datos
X = df.drop('Churn',axis=1)
y=df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression())
])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))