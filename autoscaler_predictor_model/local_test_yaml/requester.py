import requests
import time

# Укажите URL вашего сервера Nginx
url = 'http://localhost:8080'  # Замените на свой URL

while True:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Получен ответ 200: Hello, World!")
        else:
            print(f"Ошибка: Получен статус {response.status_code}")
    except Exception as e:
        print(f"Ошибка при запросе: {e}")
    
    time.sleep(5)  # Пауза 5 секунд
