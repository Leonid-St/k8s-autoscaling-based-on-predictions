import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

# Загружаем данные из CSV файла
df = pd.read_csv('../../node_data_cpu_memory_130592_623a030d0392.csv', parse_dates=['timestamp'])


# Создаем дополнительные признаки, учитывающие временную зависимость
def create_lag_features(df, lags=[1, 2, 3, 4]):
    for lag in lags:
        df[f'cpu_lag_{lag}'] = df['cpu'].shift(lag)
        df[f'memory_lag_{lag}'] = df['memory'].shift(lag)
    df = df.dropna()  # Удаляем строки с пропущенными значениями (первые строки, где нет лагов)
    return df


# Применяем создание лагов
df = create_lag_features(df)

# Определяем целевую переменную (target) и признаки (features)
X = df.drop(columns=['timestamp', 'cpu', 'memory'])
y_cpu = df['cpu']
y_memory = df['memory']

# Разделяем данные на обучающие и тестовые
X_train, X_test, y_train_cpu, y_test_cpu = train_test_split(X, y_cpu, test_size=0.5, shuffle=False)
X_train_memory, X_test_memory,  y_train_memory, y_test_memory = train_test_split(X, y_memory, test_size=0.5, shuffle=False)

# Создаем и обучаем модель XGBoost для CPU
model_cpu = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
model_cpu.fit(X_train, y_train_cpu)

# Создаем и обучаем модель XGBoost для Memory
model_memory = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
model_memory.fit(X_train_memory, y_train_memory)

# Делаем предсказания
y_pred_cpu = model_cpu.predict(X_test)
y_pred_memory = model_memory.predict(X_test_memory)


# Проверка размеров данных
print(f"Размер y_test_cpu: {len(y_test_cpu)}")
print(f"Размер y_pred_cpu: {len(y_pred_cpu)}")

# Проверка размеров данных
print(f"Размер y_test_memory: {len(y_test_memory)}")
print(f"Размер y_pred_memory: {len(y_pred_memory)}")

# Если размеры не совпадают, выравниваем их
min_length = min(len(y_test_memory), len(y_pred_memory))
y_test_memory = y_test_memory[:min_length]
y_pred_memory = y_pred_memory[:min_length]

# Если размеры не совпадают, выравниваем их
min_length = min(len(y_test_cpu), len(y_pred_cpu))
y_test_cpu = y_test_cpu[:min_length]
y_pred_cpu = y_pred_cpu[:min_length]


# Вычисляем среднеквадратичную ошибку (MSE) для CPU и Memory
mse_cpu = mean_squared_error(y_test_cpu, y_pred_cpu)
mse_memory = mean_squared_error(y_test_memory, y_pred_memory)

# Для получения RMSE, берем квадратный корень из MSE
rmse_cpu = np.sqrt(mse_cpu)
rmse_memory = np.sqrt(mse_memory)

# на синтетических данных
#Среднеквадратичная ошибка для CPU (RMSE): 17.6895
#Среднеквадратичная ошибка для Memory (RMSE): 7.7266

#На реальных данных
# Среднеквадратичная ошибка для CPU (RMSE): 13.2942
# Среднеквадратичная ошибка для Memory (RMSE): 0.3310
# Выводим результаты


# Оценка модели - RMSE и MAE
from sklearn.metrics import mean_squared_error,mean_absolute_error

mae_cpu = mean_absolute_error(y_test_cpu, y_pred_cpu)

mae_memory = mean_absolute_error(y_test_memory, y_pred_memory)

print(f"Среднеквадратичная ошибка для CPU: {rmse_cpu:.4f}")
print(f"Средняя абсолютная ошибка для CPU: {mae_cpu:.4f}")
print(f"Среднеквадратичная ошибка для Memory: {rmse_memory:.4f}")
print(f"Средняя абсолютная ошибка для Memory: {mae_memory:.4f}")



# Строим графики для CPU
plt.figure(figsize=(12, 6))

# График для CPU
plt.subplot(2, 1, 1)
plt.plot(df['timestamp'][-len(y_test_cpu):], y_test_cpu, label='Реальные данные CPU', color='blue')
plt.plot(df['timestamp'][-len(y_test_cpu):], y_pred_cpu, label='Предсказания CPU', color='red', linestyle='dashed')
plt.xlabel('Время')
plt.ylabel('Нагрузка CPU (%)')
plt.title('Реальные данные vs Предсказания (CPU)')
plt.legend()

# График для Memory
plt.subplot(2, 1, 2)
plt.plot(df['timestamp'][-len(y_test_memory):], y_test_memory, label='Реальные данные Memory', color='blue')
plt.plot(df['timestamp'][-len(y_test_memory):], y_pred_memory, label='Предсказания Memory', color='red',
         linestyle='dashed')
plt.xlabel('Время')
plt.ylabel('Нагрузка Memory (%)')
plt.title('Реальные данные vs Предсказания (Memory)')
plt.legend()

plt.tight_layout()
plt.show()
