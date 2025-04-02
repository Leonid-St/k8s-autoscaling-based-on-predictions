import json
import requests
from datetime import datetime

# Первый запрос для получения UUID
url = "https://vmselect-infra.p.ecnl.ru/select/0/prometheus/api/v1/query_range"
payload = {
    "query": 'libvirt_domain_info_vstate{project_name=~"^('
             '255|530|9117|10725|11618|11625|20401|21357|99464|111519|128696|130180|130198|130793|131137|133229'
             '|133884|135227|137477|139849|140888|140890|141279|141888|110143|130572|134762|134767|135562|136023'
             '|137023|137541|137719|138040|138174|138477|140332|141496|141567|141926|142739)_.*"}'
}

# Выполняем запрос
response = requests.post(url, json=payload)
data = response.json()

# Создаем map для уникальных UUID
uuid_map = {}

# Обрабатываем результаты
if data["status"] == "success":
    for result in data["data"]["result"]:
        uuid = result["metric"]["uuid"]
        uuid_map[uuid] = []


# Функция для преобразования timestamp в формат для Prophet
def convert_timestamp(ts):
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


# Проходим по всем UUID из карты
for uuid in uuid_map:
    # Формируем тело запроса
    payload = {
        "query": f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{uuid}"}}[1m]) * 100 / '
                 f'libvirt_domain_info_virtual_cpus{{uuid="{uuid}"}}'
    }

    # Выполняем запрос
    response = requests.post(url, json=payload)
    data = response.json()

    # Обрабатываем ответ
    if data["status"] == "success":
        for result in data["data"]["result"]:
            # Создаем список точек для временного ряда
            for value in result["values"]:
                timestamp = int(value[0])
                metric_value = float(value[1])
                # Преобразуем timestamp и добавляем точку в ряд
                uuid_map[uuid].append({
                    "ds": convert_timestamp(timestamp),  # Время в формате для Prophet
                    "y": metric_value  # Значение метрики
                })

# Выводим результат
for uuid, series in uuid_map.items():
    print(f"UUID: {uuid}")
    for point in series:
        print(f"  {point['ds']} - {point['y']}")

# Экспортируем данные в файл
with open('uuid_data.json', 'w') as f:
    json.dump(uuid_map, f, indent=4)

print("Данные успешно экспортированы в файл uuid_data.json")
