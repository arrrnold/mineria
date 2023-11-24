import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# clasificadores
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample

# evaluadores
from imblearn.under_sampling import RandomUnderSampler

# importar el dataset y hacerlo dataframe
# data = pd.read_csv("/kaggle/input/email-spam-classification-dataset/combined_data.csv")
data = pd.read_csv("../../Datasets/combined_data.csv")
data = pd.DataFrame(data)

# columnas del dataset
label = data['label']
text = data['text']

# verificar datos nulos
numero_de_valores_nulos = data.isnull().sum()
print("número de valores nulos: \n", numero_de_valores_nulos)

# media mediana y moda
print("media: \n", label.mean())
print("mediana: \n", label.median())
print("moda: \n", label.mode())

# varianza y desv est.
print("varianza: \n", label.var())
print("desviacion estándar: \n", label.std())

# cuartiles
cuartiles = label.quantile([0.25, 0.5, 0.75])
print(cuartiles)

# correos spam vs no spam
spam = data[label == 1]
no_spam = data[label == 0]

# cantidad de correos
cantidad_spam = len(spam)
cantidad_no_spam = len(no_spam)

print("número de correos spam: \n", cantidad_spam)
print("número de correos NO spam: \n", cantidad_no_spam)
print("correos totales", cantidad_spam + cantidad_no_spam)

# correlacion entre label y longitud de correo
correlacion_pearson = label.corr(text.str.len(), 'pearson')
correlacion_spearman = label.corr(text.str.len(), 'kendall')
correlacion_kendall = label.corr(text.str.len(), 'spearman')

print(correlacion_pearson)
print(correlacion_kendall)
print(correlacion_spearman)

# Crear un objeto CountVectorizer
vectorizer = CountVectorizer()

# Aplicar el vectorizador al texto de los correos electrónicos
text_vectorized = vectorizer.fit_transform(data['text'])

# text_vectorized ahora contiene la representación vectorizada de los correos electrónicos
# vectorizar los datos
vectorizer = CountVectorizer()
text_vectorizado = vectorizer.fit_transform(data['text'])

# dividir el dataset en prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(text_vectorizado, label, test_size=0.2, random_state=42)

# Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naive Bayes Accuracy:", nb_accuracy)
print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))

# validacion usando método de retención
X_train, X_test, y_train, y_test = train_test_split(text_vectorizado, label, test_size=0.2, random_state=42)
# validacion usando validación cruzada
scores = cross_val_score(nb_classifier, text_vectorizado, label, cv=5)
print("validación cruzada: \n", scores)

# validacion usando bootstrap
iteraciones = 100
bootstrap_scores = []

for _ in range(iteraciones):
    X_boot, y_boot = resample(text_vectorizado, label)
    nb_classifier.fit(X_boot, y_boot)
    bootstrap_predictions = nb_classifier.predict(X_test)
    bootstrap_accuracy = accuracy_score(y_test, bootstrap_predictions)
    bootstrap_scores.append(bootstrap_accuracy)
print("validación usando bootstrap: \n", bootstrap_scores)

# validacion usando submuestreo aleatorio
sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(text_vectorizado, label)

X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
nb_classifier.fit(X_train_resampled, y_train_resampled)
nb_predictions_resampled = nb_classifier.predict(X_test_resampled)

# calcular precision
nb_accuracy_resampled = accuracy_score(y_test_resampled, nb_predictions_resampled)
print("Naive Bayes Accuracy (Submuestreo Aleatorio):", nb_accuracy_resampled)
#informe
print("Classification Report (Submuestreo Aleatorio):\n", classification_report(y_test_resampled, nb_predictions_resampled))
# matriz de confusion
conf_matrix_resampled = confusion_matrix(y_test_resampled, nb_predictions_resampled)
print("Matriz de Confusión (Submuestreo Aleatorio):\n", conf_matrix_resampled)

# Devolver el resultado de la evaluación
resultados_evaluacion = {
    'Accuracy': nb_accuracy_resampled,
    'Classification Report': classification_report(y_test_resampled, nb_predictions_resampled),
    'Confusion Matrix': conf_matrix_resampled
}

# generar matriz de confusion
conf_matrix_nb = confusion_matrix(y_test, nb_predictions)

# imprimir matriz en consola
print("matriz de confusion: \n",conf_matrix_nb)

# graficar la matriz
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['No Spam', 'Spam'],
            yticklabels=['No Spam', 'Spam'])
plt.title('Matriz de Confusión - Naive Bayes')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.show()

# # SVM
# svm_classifier = SVC()
# svm_classifier.fit(X_train, y_train)
# svm_predictions = svm_classifier.predict(X_test)
# svm_accuracy = accuracy_score(y_test, svm_predictions)
# print("SVM Accuracy:", svm_accuracy)
# print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))
#
# # Árboles de Decisión
# dt_classifier = DecisionTreeClassifier()
# dt_classifier.fit(X_train, y_train)
# dt_predictions = dt_classifier.predict(X_test)
# dt_accuracy = accuracy_score(y_test, dt_predictions)
# print("Decision Tree Accuracy:", dt_accuracy)
# print("Decision Tree Classification Report:\n", classification_report(y_test, dt_predictions))
