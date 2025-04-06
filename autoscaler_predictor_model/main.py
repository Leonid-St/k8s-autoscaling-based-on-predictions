from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO

from sklearn.exceptions import NotFittedError
from datetime import timedelta, datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from configparser import ConfigParser
from apscheduler.schedulers.background import BackgroundScheduler
from prometheus_client import PrometheusDataCollector
import os

from utils.file_processor import process_uploaded_file
from models.polynomial import PolynomialModel
from models.xgboost import XGBoostModel
from models.sarima import SARIMAModel
from utils.cluster_metrics import NodeMetrics
from utils.metrics_comparator import MetricsComparator
from utils.metrics_storage import MetricsStorage


from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from io import StringIO
from typing import Optional

from config import EnvConfig
from storage.storage_factory import StorageFactory
from services.prediction_service import PredictionService
from services.error_calculation_service import ErrorCalculationService

app = FastAPI()

MODEL_PATH = '/app/models/'


# Global variable to store the model
cpu_model = None
memory_model = None
requests_model = None

# Добавить глобальные переменные
historical_data = pd.DataFrame()
DATA_RETENTION = timedelta(hours=4)  # Храним 4 часа данных

# Добавляем глобальные переменные
PREDICTION_FILES = {
    'polynomial': './predictions/latest_prediction_polynomial.parquet',
    'xgboost': './predictions/latest_prediction_xgboost.parquet',
    'sarima': './predictions/latest_prediction_sarima.parquet'
}

os.makedirs('./predictions', exist_ok=True)

# Инициализация конфигурации
config = ConfigParser()
config.read('config.ini')

# CPU: container_cpu_usage_seconds_total
# Memory: container_memory_working_set_bytes
# Для автоскейлинга приложений: http_requests_total

# Добавить в конфиг
PROMETHEUS_URL = config.get('PROMETHEUS', 'Url')
CPU_QUERY = config.get('PROMETHEUS', 'CpuQuery')
MEMORY_QUERY = config.get('PROMETHEUS', 'MemoryQuery')

# Инициализация коллектора
prom_collector = PrometheusDataCollector(PROMETHEUS_URL, CPU_QUERY, MEMORY_QUERY)

# Инициализация метрик кластера
cluster_metrics = NodeMetrics(PROMETHEUS_URL)

# Инициализация моделей
models = {
    'polynomial': PolynomialModel(),
    'xgboost': XGBoostModel(cluster_metrics),
    'sarima': SARIMAModel(data_retention='4H')
}

# Инициализация после других глобальных переменных
metrics_comparator = MetricsComparator(data_retention='2H')  # Например, для хранения данных за последние 2 часа
metrics_storage = MetricsStorage(storage_path='./metrics_data', retention_period=timedelta(days=365))

# Инициализация конфигурации
env_config = EnvConfig()

# Инициализация хранилища
storage = StorageFactory.create_storage({
    'STORAGE_TYPE': os.getenv('STORAGE_TYPE'),
    'POSTGRES_DB_NAME': os.getenv('POSTGRES_DB_NAME'),
    'POSTGRES_USER': os.getenv('POSTGRES_USER'),
    'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD'),
    'POSTGRES_HOST': os.getenv('POSTGRES_HOST'),
    'POSTGRES_PORT': os.getenv('POSTGRES_PORT'),
    'INFLUXDB_URL': os.getenv('INFLUXDB_URL'),
    'INFLUXDB_TOKEN': os.getenv('INFLUXDB_TOKEN'),
    'INFLUXDB_ORG': os.getenv('INFLUXDB_ORG'),
    'INFLUXDB_BUCKET': os.getenv('INFLUXDB_BUCKET')
})

# Инициализация сервисов
prediction_service = PredictionService(models['xgboost'], storage)
error_service = ErrorCalculationService(storage)

# Задача для сбора метрик
async def collect_metrics():
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=1)
    cpu_data = await prom_collector.get_cpu_metrics(env_config.uuid_node, start_time, end_time)
    if not cpu_data.empty:
        metrics_storage.save_metrics(cpu_data)

# Задача для обучения модели
async def train_model():
    await model_trainer.train_model()

# Планировщик
scheduler = AsyncIOScheduler()
scheduler.add_job(collect_metrics, 'interval', minutes=1)
scheduler.add_job(train_model, 'interval', minutes=1)
scheduler.start()

def handle_exceptions(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except NotFittedError as e:
            return jsonify({'error': f'Model not trained: {str(e)}'}), 412
        except Exception as e:
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500

    wrapper.__name__ = f.__name__
    return wrapper


def add_time_features(df):
    df = df.copy()
    df['minutes_since_midnight'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    return df


@app.post("/fit/{model_type}")
async def fit_model(model_type: str, file: UploadFile = File(...)):
    try:
        # Обработка файла и обучение модели
        df = process_uploaded_file(file)
        
        if model_type == 'polynomial':
            if 'minutes_since_midnight' not in df.columns:
                df = add_time_features(df)
            models[model_type].fit(df[['minutes_since_midnight']], df[['minutes_since_midnight']])
        elif model_type == 'xgboost':
            models[model_type].fit(df)
        elif model_type == 'sarima':
            models[model_type].update_data(df)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        # Создаем предсказание и сохраняем его
        prediction_timestamp = datetime.now() + timedelta(minutes=5)
        prediction = models[model_type].predict(prediction_timestamp)

        prediction_df = pd.DataFrame({
            'timestamp': [prediction_timestamp],
            'cpu': [prediction['cpu']],
            'memory': [prediction.get('memory', 0)]  # Для моделей, которые не предсказывают memory
        })
        prediction_df.to_csv(PREDICTION_FILES[model_type], index=False)

        return {"status": f"{model_type} model updated"}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get('/metrics/{model_type}/cpu')
async def metrics_cpu(model_type: str):
    try:
        prediction = models[model_type].predict(pd.Timestamp.now())
        max_cpu = float(config.get('DEFAULT', 'MaxCpu'))
        cpu_metric = prediction['cpu'] / max_cpu * 100

        return JSONResponse({
            'value': round(cpu_metric, 2),
            'timestamp': pd.Timestamp.now().isoformat()
        })
    except KeyError:
        raise HTTPException(status_code=400, detail='Invalid model type')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/metrics/{model_type}/memory')
async def metrics_memory(model_type: str):
    try:
        prediction = models[model_type].predict(pd.Timestamp.now())
        max_mem = float(config.get('DEFAULT', 'MaxMem'))
        mem_metric = prediction['memory'] / max_mem * 100

        return JSONResponse({
            'value': round(mem_metric, 2),
            'timestamp': pd.Timestamp.now().isoformat()
        })
    except KeyError:
        raise HTTPException(status_code=400, detail='Invalid model type')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# def scaling_decision(current_cpu_usage):
#     scale_up_threshold = config.getfloat('AUTOSCALER', 'ScaleUpThreshold')
#     scale_down_threshold = config.getfloat('AUTOSCALER', 'ScaleDownThreshold')
#     max_nodes = config.getint('AUTOSCALER', 'MaxNodes')
#     min_nodes = config.getint('AUTOSCALER', 'MinNodes')
#     scale_step = config.getint('AUTOSCALER', 'ScaleStep')
#     # Получаем текущее количество нод
#     current_nodes = cluster_metrics.get_cluster_state()['total_nodes']
#     # Логика принятия решения о масштабировании
#     if current_cpu_usage > scale_up_threshold:
#         return min(max_nodes, current_nodes + scale_step)
#     elif current_cpu_usage < scale_down_threshold:
#         return max(min_nodes, current_nodes - scale_step)
#     return current_nodes


# # Добавьте новый endpoint для получения сравнения
# @app.get('/metrics/comparison')
# def get_comparison():
#     node = request.args.get('node')
#     comparison = metrics_comparator.get_comparison(node)
#     return jsonify(comparison.to_dict(orient='records'))


# @app.get('/metrics/errors')
# def get_errors():
#     try:
#         start_date = pd.to_datetime(request.args.get('start_date'))
#         end_date = pd.to_datetime(request.args.get('end_date'))

#         errors = metrics_storage.get_errors(start_date, end_date)
#         return jsonify(errors.to_dict(orient='records'))
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# Добавляем новый endpoint для анализа точности
@app.get('/metrics/accuracy/{model_type}')
def get_accuracy(
    model_type: str, 
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    try:
        if model_type not in models:
            return JSONResponse(
                content={'error': f'Invalid model type: {model_type}'},
                status_code=400
            )

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        accuracy_data = metrics_storage.calculate_accuracy_over_time(start_date, end_date, model_type)
        if accuracy_data.empty:
            return JSONResponse(
                content={'error': f'No accuracy data available for {model_type} in the specified period'},
                status_code=404
            )

        return JSONResponse(content=accuracy_data.to_dict(orient='records'))
    except ValueError as e:
        return JSONResponse(content={'error': str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)


# Endpoint для получения последнего предсказания
@app.get('/predict/latest/{model_type}')
def get_latest_prediction(model_type: str):
    try:
        if model_type not in PREDICTION_FILES:
            return JSONResponse(
                content={'error': f'Invalid model type: {model_type}'},
                status_code=400
            )

        prediction_file = PREDICTION_FILES[model_type]
        if not os.path.exists(prediction_file):
            return JSONResponse(
                content={'error': f'No predictions available for {model_type}. Model may not have been trained yet.'}, 
                status_code=404
            )

        prediction = pd.read_csv(prediction_file)
        prediction['timestamp'] = pd.to_datetime(prediction['timestamp'])  # Преобразуем в datetime
        latest_prediction = prediction.iloc[-1].to_dict()
        latest_prediction['timestamp'] = latest_prediction['timestamp'].isoformat()  # Преобразуем в строку

        return JSONResponse(content=latest_prediction)
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)


# Пример использования
async def collect_and_predict():
    # Сбор данных
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=1)
    cpu_data = await prom_collector.get_cpu_metrics(env_config.uuid_node, start_time, end_time)
    
    if not cpu_data.empty:
        # Сохранение реальных данных
        storage.save_actual(end_time, env_config.uuid_node, {'cpu': cpu_data['value'].mean()})
        
        # Создание предсказания
        prediction = prediction_service.make_prediction(end_time, env_config.uuid_node)
        
        # Расчет ошибок
        error_service.calculate_errors(start_time, end_time, env_config.uuid_node, 'xgboost')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
