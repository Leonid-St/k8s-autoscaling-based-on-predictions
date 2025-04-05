# import json
# import requests
# from datetime import datetime
# import warnings
# from urllib3.exceptions import NotOpenSSLWarning
#
# # Подавляем предупреждение NotOpenSSLWarning
# warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
#
# # URL для запросов
# url = "https://vmselect-infra.p.ecnl.ru/select/0/prometheus/api/v1/query_range"
#
# # Первый запрос для получения UUID
# query = 'libvirt_domain_info_vstate{project_name=~"^(' \
#         '255|530|9117|10725|11618|11625|20401|21357|99464|111519|128696|130180|130198|130793|131137|133229' \
#         '|133884|135227|137477|139849|140888|140890|141279|141888|110143|130572|134762|134767|135562|136023' \
#         '|137023|137541|137719|138040|138174|138477|140332|141496|141567|141926|142739)_.*"}'
#
# # Тело запроса
# payload = {
#     "query": query
# }
#
# # Выполняем запрос
# response = requests.post(url, data=payload)
# data = response.json()
#
# # Создаем map для уникальных UUID
# uuid_map = {}
#
# # Обрабатываем результаты
# if data["status"] == "success":
#     for result in data["data"]["result"]:
#         uuid = result["metric"]["uuid"]
#         uuid_map[uuid] = []
#
# # Функция для преобразования timestamp в формат для Prophet
# def convert_timestamp(ts):
#     return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#
# # Проходим по всем UUID из карты
# for uuid in uuid_map:
#     # Формируем тело запроса
#     query = f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{uuid}"}}[1m]) * 100 / ' \
#             f'libvirt_domain_info_virtual_cpus{{uuid="{uuid}"}}'
#     payload = {
#         "query": query
#     }
#
#     # Выполняем запрос
#     response = requests.post(url, data=payload)
#     data = response.json()
#
#     # Обрабатываем ответ
#     if data["status"] == "success":
#         for result in data["data"]["result"]:
#             # Создаем список точек для временного ряда
#             for value in result["values"]:
#                 timestamp = int(value[0])
#                 metric_value = float(value[1])
#                 # Преобразуем timestamp и добавляем точку в ряд
#                 uuid_map[uuid].append({
#                     "ds": convert_timestamp(timestamp),  # Время в формате для Prophet
#                     "y": metric_value  # Значение метрики
#                 })
#
# # Выводим результат
# for uuid, series in uuid_map.items():
#     print(f"UUID: {uuid}")
#     for point in series:
#         print(f"  {point['ds']} - {point['y']}")
#
# # Экспортируем данные в файл
# with open('uuid_data.json', 'w') as f:
#     json.dump(uuid_map, f, indent=4)
#
# print("Данные успешно экспортированы в файл uuid_data.json")
#
#
#

import json
import requests
from datetime import datetime

# URL для запросов
url = "https://vmselect-infra.p.ecnl.ru/select/0/prometheus/api/v1/query_range"

# Первый запрос для получения UUID и node
query = 'libvirt_domain_info_vstate{project_name=~"^(' \
        '255|530|9117|10725|11618|11625|20401|21357|99464|111519|128696|130180|130198|130793|131137|133229' \
        '|133884|135227|137477|139849|140888|140890|141279|141888|110143|130572|134762|134767|135562|136023' \
        '|137023|137541|137719|138040|138174|138477|140332|141496|141567|141926|142739)_.*"}'

# Тело запроса
payload = {
    "query": query
}

try:
    # Выполняем запрос
    print("Пытаюсь выполнить запрос...")
    response = requests.post(url, data=payload)
    response.raise_for_status()  # Проверка на ошибки HTTP
    data = response.json()
    print("Запрос выполнен успешно.")
except requests.exceptions.RequestException as e:
    print(f"Ошибка при выполнении запроса: {e}")
    exit(1)

# Создаем map для хранения uuid и соответствующего node
uuid_node_map = {}

# Обрабатываем результаты
if data["status"] == "success":
    for result in data["data"]["result"]:
        uuid = result["metric"]["uuid"]
        node = result["metric"]["node"]
        uuid_node_map[uuid] = node

# Создаем map для группировки данных по node
node_data_map = {}

# Функция для преобразования timestamp в формат для Prophet
def convert_timestamp(ts):
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# Проходим по всем UUID из карты
for uuid, node in uuid_node_map.items():
    # Формируем тело запроса
    query = f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{uuid}"}}[1y]) * 100 / ' \
            f'libvirt_domain_info_virtual_cpus{{uuid="{uuid}"}}'
    payload = {
        "query": query
    }

    try:
        # Выполняем запрос
        print(f"Выполняю запрос для UUID: {uuid} (Node: {node})...")
        response = requests.post(url, data=payload)
        response.raise_for_status()  # Проверка на ошибки HTTP
        data = response.json()
        print("Запрос выполнен успешно.")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при выполнении запроса для UUID {uuid}: {e}")
        continue

    # Обрабатываем ответ
    if data["status"] == "success":
        for result in data["data"]["result"]:
            # Если node еще не в карте, добавляем его
            if node not in node_data_map:
                node_data_map[node] = []
            # Добавляем все значения (timestamp + метрика) в массив для этого node
            for value in result["values"]:
                timestamp = int(value[0])
                metric_value = float(value[1])
                node_data_map[node].append({
                    "ds": convert_timestamp(timestamp),  # Время в формате для Prophet
                    "y": metric_value  # Значение метрики
                })

# Выводим результат
for node, series in node_data_map.items():
    print(f"Node: {node}")
    for point in series:
        print(f"  {point['ds']} - {point['y']}")

# Экспортируем данные в файл
with open('node_data.json', 'w') as f:
    json.dump(node_data_map, f, indent=4)

print("Данные успешно экспортированы в файл node_data.json")
