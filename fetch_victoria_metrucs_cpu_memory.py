import json
import requests
from datetime import datetime, timedelta
import warnings
from urllib3.exceptions import NotOpenSSLWarning

# Подавляем предупреждение NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# URL для запросов
url = "https://vmselect-infra.p.ecnl.ru/select/0/prometheus/api/v1/query_range"

# Указываем UUID для запроса
uuid = "6205fe93-5567-4f8a-ae29-8fc6fb34f213"

# Создаем map для группировки данных по node
node_data_map = {}


# Функция для преобразования timestamp в формат для Prophet
def convert_timestamp(ts):
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


# Указываем временной диапазон: последний год
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=365)

# Шаг для разбиения временного диапазона (15 дней)
time_delta = timedelta(days=15)


# Функция для сброса данных в файл
def dump_data_to_file(node_data_map, filename):
    # Сортируем данные по времени для каждого node
    for node in node_data_map:
        node_data_map[node].sort(key=lambda x: x["ds"])

    # Открываем файл для добавления данных
    with open(filename, 'a') as f:
        for node, series in node_data_map.items():
            for point in series:
                f.write(json.dumps({
                    "node": node,
                    "ds": point["ds"],
                    "cpu": point["cpu"],
                    "memory": point["memory"]
                }) + "\n")


# Указываем node (если она известна) или оставляем пустой строкой
node = ""  # Замените на известное значение node, если оно есть

current_start = start_time
while current_start < end_time:
    current_end = min(current_start + time_delta, end_time)

    # Формируем запросы для CPU и memory
    cpu_query = f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{uuid}"}}[1m]) * 100 / ' \
                f'libvirt_domain_info_virtual_cpus{{uuid="{uuid}"}}'
    memory_query = f'libvirt_domain_memory_stats_used_percent{{uuid="{uuid}"}}'

    # Выполняем запрос для CPU
    try:
        print(f"Выполняю запрос CPU для UUID: {uuid} с {current_start} по {current_end}...")
        cpu_response = requests.get(url, params={
            "query": cpu_query,
            "start": current_start.timestamp(),
            "end": current_end.timestamp(),
            "step": "1m"
        })
        cpu_response.raise_for_status()
        cpu_data = cpu_response.json()
        print("Запрос CPU выполнен успешно.")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при выполнении запроса CPU для UUID {uuid}: {e}")
        current_start = current_end
        continue

    # Выполняем запрос для memory
    try:
        print(f"Выполняю запрос Memory для UUID: {uuid} с {current_start} по {current_end}...")
        memory_response = requests.get(url, params={
            "query": memory_query,
            "start": current_start.timestamp(),
            "end": current_end.timestamp(),
            "step": "1m"
        })
        memory_response.raise_for_status()
        memory_data = memory_response.json()
        print("Запрос Memory выполнен успешно.")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при выполнении запроса Memory для UUID {uuid}: {e}")
        current_start = current_end
        continue

    # Обрабатываем ответы
    if cpu_data["status"] == "success" and memory_data["status"] == "success":
        # Создаем временные карты для CPU и memory
        cpu_map = {int(value[0]): float(value[1]) for result in cpu_data["data"]["result"] for value in
                   result["values"]}
        memory_map = {int(value[0]): float(value[1]) for result in memory_data["data"]["result"] for value in
                      result["values"]}

        # Если node еще не в карте, добавляем его
        if node not in node_data_map:
            node_data_map[node] = []

        # Собираем данные по временным меткам
        for timestamp in set(cpu_map.keys()).union(memory_map.keys()):
            node_data_map[node].append({
                "ds": convert_timestamp(timestamp),
                "cpu": cpu_map.get(timestamp, None),  # None, если данных нет
                "memory": memory_map.get(timestamp, None)  # None, если данных нет
            })

    # Переходим к следующему временному интервалу
    current_start = current_end

# Сбрасываем данные в файл
dump_data_to_file(node_data_map, 'node_data_cpu_memory_9125_6205fe93-5567-4f8a-ae29-8fc6fb34f213.json')

print("Данные успешно экспортированы в файл node_data.jsonl")
