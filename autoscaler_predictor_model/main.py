import asyncio
import json
import os

from fastapi import FastAPI, Depends, HTTPException, status
import requests
from datetime import datetime, timedelta
import warnings
from urllib3.exceptions import NotOpenSSLWarning

from models.polynomial import PolynomialModel
from metrics.metrics_collector import MetricsCollector
from config import EnvConfig
from metrics.metrics_fetcher import MetricsFetcher
from metrics.metrics_fetcher_victoria import VictoriaMetricsFetcher
from models.sarima import SARIMAModel
from models.xgboost import XGBoostModel
from services.predictor_service import PredictorService
from services.teacher_service import TeacherService
from storage.storage_factory import StorageFactory
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from storage.storage_service import StorageService
import uvicorn
from contextlib import asynccontextmanager

from services.comparator_service import ComparatorService
from fastapi.responses import JSONResponse
from services.scaler_service import ScalerService

# MODEL_PATH = '/app/models/'


# # Global variable to store the model
# cpu_model = None
# memory_model = None
# requests_model = None

# # Добавить глобальные переменные
# historical_data = pd.DataFrame()
# DATA_RETENTION = timedelta(hours=4)  # Храним 4 часа данных

# # Добавляем глобальные переменные
# PREDICTION_FILES = {
#     'polynomial': './predictions/latest_prediction_polynomial.parquet',
#     'xgboost': './predictions/latest_prediction_xgboost.parquet',
#     'sarima': './predictions/latest_prediction_sarima.parquet'
# }

# os.makedirs('./predictions', exist_ok=True)


# # Задача для сбора метрик
# async def collect_metrics():
#     end_time = datetime.now()
#     start_time = end_time - timedelta(minutes=1)
#     cpu_data = await prom_collector.get_cpu_metrics(env_config.uuid_node, start_time, end_time)
#     if not cpu_data.empty:
#         metrics_storage.save_metrics(cpu_data)


# def add_time_features(df):
#     df = df.copy()
#     df['minutes_since_midnight'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
#     return df


# @app.post("/fit/{model_type}")
# async def fit_model(model_type: str, file: UploadFile = File(...)):
#     return {"status": f"{200} model updated"}
# try:
#     # Обработка файла и обучение модели
#     df = process_uploaded_file(file)

#     if model_type == 'polynomial':
#         if 'minutes_since_midnight' not in df.columns:
#             df = add_time_features(df)
#         models[model_type].fit(df[['minutes_since_midnight']], df[['minutes_since_midnight']])
#     elif model_type == 'xgboost':
#         models[model_type].fit(df)
#     elif model_type == 'sarima':
#         models[model_type].update_data(df)
#     else:
#         raise HTTPException(status_code=400, detail="Invalid model type")

#     # Создаем предсказание и сохраняем его
#     prediction_timestamp = datetime.now() + timedelta(minutes=5)
#     prediction = models[model_type].predict(prediction_timestamp)

#     prediction_df = pd.DataFrame({
#         'timestamp': [prediction_timestamp],
#         'cpu': [prediction['cpu']],
#         'memory': [prediction.get('memory', 0)]  # Для моделей, которые не предсказывают memory
#     })
#     prediction_df.to_csv(PREDICTION_FILES[model_type], index=False)

#     return {"status": f"{model_type} model updated"}

# except KeyError as e:
#     raise HTTPException(status_code=400, detail=str(e))
# except Exception as e:
#     raise HTTPException(status_code=500, detail=str(e))


# @app.get("/predict/<model_type>")
# @handle_exceptions
# def predict(model_type):
#     try:
#         timestamp_str = request.args.get('timestamp')
#         if not timestamp_str:
#             return jsonify({'error': 'Missing timestamp'}), 400

#         timestamp = pd.to_datetime(timestamp_str)
#         result = models[model_type].predict(timestamp)

#         # Сбор и сохранение предсказанных данных
#         cluster_state = cluster_metrics.get_cluster_state()
#         for node, metrics in cluster_state['nodes'].items():
#             metrics_storage.save_metrics(
#                 timestamp=timestamp,
#                 node=node,
#                 cpu_actual=None,
#                 cpu_predicted=result['cpu'],
#                 model_type=model_type  # Указываем тип модели
#             )

#         return jsonify({
#             'model': model_type,
#             'timestamp': timestamp.isoformat(),
#             **result
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# @app.post('/forecast')
# def forecast():
#     # Check if the request contains JSON data
#     if request.is_json:
#         data = request.get_json()
#         df = pd.DataFrame(data)
#         df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     else:
#         # Read CSV file from the request
#         file = request.files.get('file')
#         if not file:
#             return jsonify({'error': 'No file provided'}), 400
#         # Convert file to DataFrame
#         df = pd.read_csv(StringIO(file.read().decode('utf-8')))

#     # Validate DataFrame format and presence of timestamp
#     if 'timestamp' not in df.columns:
#         return jsonify({'error': 'CSV/JSON must contain timestamp column'}), 400

#     # Convert timestamps to DateTime and sort
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df.set_index('timestamp', inplace=True)
#     df.sort_index(inplace=True)

#     # Predict for the next 15 minutes
#     last_timestamp = df.index[-1]
#     next_timestamp = last_timestamp + timedelta(minutes=15)

#     prediction = {}

#     # SARIMA models for each type of data
#     if 'cpu' in df.columns:
#         cpu_model = SARIMAX(df['cpu'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
#         cpu_results = cpu_model.fit()
#         prediction['cpu'] = cpu_results.forecast(steps=1)[0]
#     if 'memory' in df.columns:
#         memory_model = SARIMAX(df['memory'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
#         memory_results = memory_model.fit()
#         prediction['memory'] = memory_results.forecast(steps=1)[0]

#     if 'requests' in df.columns:
#         requests_model = SARIMAX(df['requests'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
#         requests_results = requests_model.fit()
#         prediction['requests'] = requests_results.forecast(steps=1)[0]

#     prediction['timestamp'] = next_timestamp.strftime('%Y-%m-%d %H:%M:%S')
#     return jsonify(prediction)


# @app.get('/metrics/<model_type>')
# def metrics(model_type):
#     try:
#         # Получаем прогноз от выбранной модели
#         prediction = models[model_type].predict(pd.Timestamp.now())

#         # Получаем максимальные значения из конфига
#         max_cpu = float(config.get('DEFAULT', 'MaxCpu'))
#         max_mem = float(config.get('DEFAULT', 'MaxMem'))

#         # Вычисляем комбинированную метрику
#         combined_metric = max(
#             prediction['cpu'] / max_cpu * 100,
#             prediction['memory'] / max_mem * 100
#         )

#         return jsonify({
#             'value': round(combined_metric, 2),
#             'timestamp': pd.Timestamp.now().isoformat()
#         })

#     except KeyError:
#         return jsonify({'error': 'Invalid model type'}), 400
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# @app.get('/metrics/{model_type}/cpu')
# async def metrics_cpu(model_type: str):
#     try:
#         prediction = models[model_type].predict(pd.Timestamp.now())
#         max_cpu = float(config.get('DEFAULT', 'MaxCpu'))
#         cpu_metric = prediction['cpu'] / max_cpu * 100

#         return JSONResponse({
#             'value': round(cpu_metric, 2),
#             'timestamp': pd.Timestamp.now().isoformat()
#         })
#     except KeyError:
#         raise HTTPException(status_code=400, detail='Invalid model type')
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get('/metrics/{model_type}/memory')
# async def metrics_memory(model_type: str):
#     try:
#         prediction = models[model_type].predict(pd.Timestamp.now())
#         max_mem = float(config.get('DEFAULT', 'MaxMem'))
#         mem_metric = prediction['memory'] / max_mem * 100

#         return JSONResponse({
#             'value': round(mem_metric, 2),
#             'timestamp': pd.Timestamp.now().isoformat()
#         })
#     except KeyError:
#         raise HTTPException(status_code=400, detail='Invalid model type')
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # Добавляем новый endpoint для анализа точности
# @app.get('/metrics/accuracy/{model_type}')
# def get_accuracy(
#     model_type: str, 
#     start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
#     end_date: str = Query(..., description="End date in YYYY-MM-DD format")
# ):
#     try:
#         if model_type not in models:
#             return JSONResponse(
#                 content={'error': f'Invalid model type: {model_type}'},
#                 status_code=400
#             )

#         start_date = pd.to_datetime(start_date)
#         end_date = pd.to_datetime(end_date)

#         accuracy_data = metrics_storage.calculate_accuracy_over_time(start_date, end_date, model_type)
#         if accuracy_data.empty:
#             return JSONResponse(
#                 content={'error': f'No accuracy data available for {model_type} in the specified period'},
#                 status_code=404
#             )

#         return JSONResponse(content=accuracy_data.to_dict(orient='records'))
#     except ValueError as e:
#         return JSONResponse(content={'error': str(e)}, status_code=400)
#     except Exception as e:
#         return JSONResponse(content={'error': str(e)}, status_code=500)


# # Endpoint для получения последнего предсказания
# @app.get('/predict/latest/{model_type}')
# def get_latest_prediction(model_type: str):
#     try:
#         if model_type not in PREDICTION_FILES:
#             return JSONResponse(
#                 content={'error': f'Invalid model type: {model_type}'},
#                 status_code=400
#             )

#         prediction_file = PREDICTION_FILES[model_type]
#         if not os.path.exists(prediction_file):
#             return JSONResponse(
#                 content={'error': f'No predictions available for {model_type}. Model may not have been trained yet.'}, 
#                 status_code=404
#             )

#         prediction = pd.read_csv(prediction_file)
#         prediction['timestamp'] = pd.to_datetime(prediction['timestamp'])  # Преобразуем в datetime
#         latest_prediction = prediction.iloc[-1].to_dict()
#         latest_prediction['timestamp'] = latest_prediction['timestamp'].isoformat()  # Преобразуем в строку

#         return JSONResponse(content=latest_prediction)
#     except Exception as e:
#         return JSONResponse(content={'error': str(e)}, status_code=500)

# class Metric:
#     timastap int
#     cpu
# class FetcherMetrics:

#     def __init__(self,*,url:str,uuid:str,storage):
#         self.url = url
#         self.uuid = uuid
#         self.storage = storage

#     def fetch_metrics_and_colletc_in_storage(self):
#         self.fetch_metrics()

#     def fetch_metric_1m(self):
#         query = f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{self.uuid}"}}[1m]) * 100 / ' \
#         f'libvirt_domain_info_virtual_cpus{{uuid="{self.uuid}"}}'
#         params = {
#             "query": query,
#             "step": "1m"  # Шаг (например, 60 секунд)
#             }

#         try:
#             # Выполняем запрос
#             print(f"Выполняю запрос для UUID: {self.uuid} с {current_start} по {current_end}...")
#             response = requests.get(self.url, params=params)
#             response.raise_for_status()  # Проверка на ошибки HTTP
#             data = response.json()
#             print("Запрос выполнен успешно.")
#         except requests.exceptions.RequestException as e:
#             print(f"Ошибка при выполнении запроса для UUID {self.uuid}: {e}")
#             current_start = current_end
#             continue

#         # Обрабатываем ответ
#         if data["status"] == "success":
#             for result in data["data"]["result"]:

#                 # Добавляем все значения (timestamp + метрика) в массив для этого node
#                 #for value in result["values"]:
#                     timestamp = int(value[0])
#                     metric_value = float(value[1])
#                     node_data_map[node].append({
#                         "ds": convert_timestamp(timestamp),  # Время в формате для Prophet
#                         "y": metric_value  # Значение метрики
#                     })


# def fetch_metrics(self):
#     # Подавляем предупреждение NotOpenSSLWarning
#     warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

#     # Создаем map для группировки данных по node
#     node_data_map = {}


#     # Функция для преобразования timestamp в формат для Prophet
#     def convert_timestamp(ts):
#         return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


#     # Указываем временной диапазон: последний год
#     end_time = datetime.utcnow()
#     start_time = end_time - timedelta(days=365)

#     # Шаг для разбиения временного диапазона (15 дней)
#     time_delta = timedelta(days=15)


#     # Функция для сброса данных в файл
#     def dump_data_to_file(node_data_map, filename):
#         # Сортируем данные по времени для каждого node
#         for node in node_data_map:
#             node_data_map[node].sort(key=lambda x: x["ds"])

#         # Открываем файл для добавления данных
#         with open(filename, 'a') as f:
#             for node, series in node_data_map.items():
#                 for point in series:
#                     f.write(json.dumps({"node": node, "ds": point["ds"], "y": point["y"]}) + "\n")


#     # Указываем node (если она известна) или оставляем пустой строкой
#     node = ""  # Замените на известное значение node, если оно есть

#     current_start = start_time
#     while current_start < end_time:
#         current_end = min(current_start + time_delta, end_time)
#         # Формируем тело запроса
#         query = f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{self.uuid}"}}[1m]) * 100 / ' \
#                 f'libvirt_domain_info_virtual_cpus{{uuid="{self.uuid}"}}'
#         params = {
#             "query": query,
#             "start": current_start.timestamp(),  # Начало периода (timestamp)
#             "end": current_end.timestamp(),  # Конец периода (timestamp)
#             "step": "1m"  # Шаг (например, 60 секунд)
#         }

#         try:
#             # Выполняем запрос
#             print(f"Выполняю запрос для UUID: {self.uuid} с {current_start} по {current_end}...")
#             response = requests.get(self.url, params=params)
#             response.raise_for_status()  # Проверка на ошибки HTTP
#             data = response.json()
#             print("Запрос выполнен успешно.")
#         except requests.exceptions.RequestException as e:
#             print(f"Ошибка при выполнении запроса для UUID {self.uuid}: {e}")
#             current_start = current_end
#             continue

#         # Обрабатываем ответ
#         if data["status"] == "success":
#             for result in data["data"]["result"]:
#                 # Если node еще не в карте, добавляем его
#                 if node not in node_data_map:
#                     node_data_map[node] = []
#                 # Добавляем все значения (timestamp + метрика) в массив для этого node
#                 for value in result["values"]:
#                     timestamp = int(value[0])
#                     metric_value = float(value[1])
#                     node_data_map[node].append({
#                         "ds": convert_timestamp(timestamp),  # Время в формате для Prophet
#                         "y": metric_value  # Значение метрики
#                     })


#         # Переходим к следующему временному интервалу
#         current_start = current_end

#     # Сбрасываем данные в файл
#     dump_data_to_file(node_data_map, 'node_data_130592_623a030d0392.json')

#     print("Данные успешно экспортированы в файл node_data.jsonl")

# Задача для сбора метрик
# async def collect_metrics():
#     end_time = datetime.now()
#     start_time = end_time - timedelta(minutes=1)
#     cpu_data = await prom_collector.get_cpu_metrics(env_config.uuid_node, start_time, end_time)
#     if not cpu_data.empty:
#         metrics_storage.save_metrics(cpu_data)


from pytz import utc

scheduler = AsyncIOScheduler(timezone=utc)



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация конфигурации
    env_config = EnvConfig()

    # 0 - Инициализация хранилища
    storage = StorageFactory.create_storage(env_config)

    # 1 - fetch metrics
    fetcher = VictoriaMetricsFetcher(env_config.url_for_metrics)

    collector = MetricsCollector(
        uuid=env_config.uuid_node,
        fetcher=fetcher,
        storage=storage,
    )

    # Инициализация моделей
    models = {
        'polynomial': PolynomialModel(),
        'xgboost': XGBoostModel(),
        'sarima': SARIMAModel(data_retention='4H')
    }

    teacher_service = TeacherService(
        model=models['xgboost'],
        storage=storage,
        node_id=env_config.uuid_node
    )

    # Регистрируем TeacherService как наблюдателя
    collector.add_observer(teacher_service.on_new_data)

    # Инициализация PredictorService
    predictor_service = PredictorService(
        model=models['xgboost'],
        storage=storage,
        node_id=env_config.uuid_node
    )

    # Регистрируем PredictorService как наблюдателя в TeacherService
    teacher_service.add_observer(predictor_service.on_model_updated)

    # Инициализация ComparatorService
    comparator_service = ComparatorService(
        storage=storage,
        node_id=env_config.uuid_node
    )

    # Регистрируем ComparatorService как наблюдателя в PredictorService
    predictor_service.add_observer(comparator_service.compare_and_save_errors)

    scheduler.add_job(collector.collect, 'interval', seconds=60)

    # Инициализация ScalerService
    scaler_service = ScalerService(
        storage=storage,
        node_id=env_config.uuid_node,
        keda_scaler_address="http://autoscaler-service:5001"
    )
    scheduler.add_job(scaler_service.scale, 'interval', minutes=1)

    scheduler.start()

    yield

    # Закрытие соединения с базой данных
    if hasattr(storage, 'conn') and not storage.conn.closed:
        storage.conn.close()
    scheduler.shutdown()


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("uvicorn.error")


    app = FastAPI(lifespan=lifespan)


async def get_predictor_service() -> PredictorService:
    env_config = EnvConfig()
    storage = StorageFactory.create_storage(env_config)
    models = {
        'xgboost': XGBoostModel(),
        'sarima': SARIMAModel(data_retention='4H'),
        'polynomial': PolynomialModel()
    }
    return PredictorService(
        model=models['xgboost'],
        storage=storage,
        node_id=env_config.uuid_node
    )


async def get_storage_service() -> StorageService:
    env_config = EnvConfig()
    return StorageFactory.create_storage(env_config)


@app.get("/keda-metrics")
async def keda_metrics(storage_service: StorageService = Depends(get_storage_service)):
    """
    Endpoint для Keda, возвращает последнее предсказанное значение CPU из базы данных.
    """
    latest_prediction = await storage_service.get_prediction(
        node=EnvConfig().uuid_node,
        timestamp=datetime.now()
    )

    if latest_prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Предсказания не найдены")

    cpu_prediction = latest_prediction.metrics.get('cpu')

    if cpu_prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CPU предсказание не найдено")

    return JSONResponse({"cpu": cpu_prediction})
