import json
import csv

# Путь к входному файлу .json и выходному файлу .csv
input_file = 'node_data_cpu_memory_130592_fa84e6f2-fb98-4304-b4cb-6c591f967446.json'
output_file = 'node_data_cpu_memory_130592_fa84e6f2-fb98-4304-b4cb-6c591f967446.csv'

# Открываем входной файл для чтения и выходной для записи
with open(input_file, 'r') as json_file, open(output_file, 'w', newline='') as csv_file:
    # Создаем writer для CSV
    csv_writer = csv.writer(csv_file)

    # Записываем заголовки столбцов
    csv_writer.writerow(['node', 'ds', 'cpu', 'memory'])

    # Читаем каждую строку из .json файла
    for line in json_file:
        # Парсим JSON-строку
        data = json.loads(line)

        # Записываем данные в CSV
        csv_writer.writerow([
            data['node'],
            data['ds'],
            data['cpu'],
            data['memory']
        ])

print(f"Данные успешно преобразованы и сохранены в {output_file}")