import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle

# 1. Cargar el dataset
df = pd.read_csv("student-mat.csv", sep=";")

# 2. Crear la variable objetivo (1 = aprobó, 0 = no aprobó)
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# 3. Seleccionar variables independientes relevantes
features = ['studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['pass']

# 4. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Entrenar el modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluar el modelo
y_pred = model.predict(X_test)
print("Evaluación del modelo:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# 7. Guardar el modelo
with open('modelo_aprobacion.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Modelo guardado como 'modelo_aprobacion.pkl'")
