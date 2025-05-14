import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Загрузка данных
df = pd.read_csv("C:\\Users\\admin\\Desktop\\labaAI\\лаба1\\venv\\traincompleted.csv")

# Разделяем данные на признаки (X) и целевую переменную (y)
X = df.drop('Transported', axis=1)
y = df['Transported']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем конвейер с масштабированием и моделью логистической регрессии
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))

# Обучаем модель
pipe.fit(X_train, y_train)

# Предсказания и оценка
y_pred = pipe.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Логистическая регрессия:")
print("Accuracy:", accuracy_log)
print("Полнота (Recall):", recall)
print("Точность классификации (Precision):", precision)
print("F1-мера:", f1)
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))