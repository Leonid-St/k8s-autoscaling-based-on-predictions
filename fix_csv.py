import pandas as pd

# Чтение CSV файла
df = pd.read_csv('node_data_cpu_memory_130592_623a030d0392.csv')

# Удаление колонки 'node' и лишних запятых
df = df.drop(columns=['node'])

# Переименование колонки 'ds' в 'timestamp'
df = df.rename(columns={'ds': 'timestamp'})

# Сохранение обработанного файла
df.to_csv('node_data_cpu_memory_130592_623a030d0392.csv', index=False)
