import pandas as pd
import numpy as np


def fill_missing_with_expanding_mean(df, columns, window=3):
    """
    Заполняет пропуски в указанных колонках, используя скользящее среднее.
    Если ближайшие значения отсутствуют, расширяет окно поиска.

    :param df: DataFrame с данными
    :param columns: Список колонок для обработки
    :param window: Максимальное окно для поиска значений
    :return: DataFrame с заполненными пропусками
    """
    for col in columns:
        # Создаем копию колонки для работы
        series = df[col].copy()

        # Находим индексы пропущенных значений
        missing_indices = series[series.isna()].index

        for idx in missing_indices:
            # Ищем ближайшие непустые значения в окне
            left = max(0, idx - window)
            right = min(len(series) - 1, idx + window)

            # Собираем все непустые значения в окне
            valid_values = series[left:right + 1].dropna()

            if not valid_values.empty:
                # Заполняем пропуск средним значением
                series[idx] = valid_values.mean()

        # Обновляем колонку в DataFrame
        df[col] = series

    return df


# Открываем CSV-файл
input_file = 'node_data_cpu_memory_130592_623a030d0392.csv'  # Укажите путь к вашему файлу
df = pd.read_csv(input_file)

# Преобразуем колонку timestamp в datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Заполняем пропуски в колонках cpu и memory
df_filled = fill_missing_with_expanding_mean(df, ['cpu', 'memory'], window=3)

# Сохраняем результат в новый файл
output_file = 'node_data_cpu_memory_130592_623a030d0392.csv'
df_filled.to_csv(output_file, index=False)

print(f"Обработанные данные сохранены в файл: {output_file}")