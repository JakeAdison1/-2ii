import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix


df = pd.read_csv("C:\\Users\\admin\\Desktop\\labaAI\\лаба1\\venv\\traincompleted.csv")

# Разделяем данные на признаки (X) и целевую переменную (y)
X = df.drop('Age', axis=1)
y = df['Age']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Линейная регрессия
lin_model = Ridge(alpha=2.0)
lin_model.fit(X_train, y_train)

# Предсказания
y_pred_test = lin_model.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
MSE = mean_squared_error(y_test, y_pred_test)
RMSE = np.sqrt(MSE)
MAE = mean_absolute_error(y_test, y_pred_test)

print(f"MSE (среднеквадартичная ошибка): {MSE:.2f}")
print(f"RMSE (корень среднеквадратичной ошибки): {RMSE:.2f}")
print(f"MAE (средняя абсолютная ошибка (по модулю))): {MAE:.2f}")