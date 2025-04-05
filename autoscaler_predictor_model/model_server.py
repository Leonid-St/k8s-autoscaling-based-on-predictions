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
from flask import Flask
from flask_swagger_ui import get_swaggerui_blueprint

MODEL_PATH = '/app/models/'

app = Flask(__name__)


# Определите путь к вашему Swagger JSON
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'  # Путь к вашему swagger.json

# Настройка Swagger UI
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Autoscaler Predictor Model API"
    }
)

# Global variable to store the model
cpu_model = None
memory_model = None
requests_model = None

# Добавить глобальные переменные
historical_data = pd.DataFrame()
DATA_RETENTION = timedelta(hours=4)  # Храним 4 часа данных

# Добавляем глобальные переменные
PREDICTION_FILE = './predictions/latest_prediction.parquet'
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


# def update_model_from_prometheus():
#     try:
#         new_data = prom_collector.get_new_data()
#         if not new_data.empty:
#             models['xgboost'].partial_fit(new_data)

#             # Сбор и сохранение реальных данных
#             for _, row in new_data.iterrows():
#                 metrics_storage.save_metrics(
#                     timestamp=row['timestamp'],
#                     node=row['node'],
#                     cpu_actual=row['cpu'],
#                     cpu_predicted=None  # Реальные данные, предсказания пока нет
#                 )

#             print(f"Model updated with {len(new_data)} new samples")
#     except Exception as e:
#         print(f"Error updating model: {str(e)}")


def update_and_predict():
    try:
        # 1. Сбор данных
        new_data = prom_collector.get_new_data()
        if new_data.empty:
            return

        # 2. Дообучение модели
        models['xgboost'].partial_fit(new_data)

        # 3. Создание предсказания
        prediction_timestamp = datetime.now() + timedelta(minutes=5)
        prediction = models['xgboost'].predict(prediction_timestamp)

        # 4. Сохранение предсказания
        prediction_df = pd.DataFrame({
            'timestamp': [prediction_timestamp],
            'cpu': [prediction['cpu']],
            'memory': [prediction['memory']]
        })
        prediction_df.to_parquet(PREDICTION_FILE, index=False)

        # 5. Расчет и сохранение ошибок
        for _, row in new_data.iterrows():
            actual_cpu = row['cpu']
            predicted_cpu = prediction['cpu']

            absolute_error = abs(actual_cpu - predicted_cpu)
            relative_error = absolute_error / actual_cpu if actual_cpu != 0 else 0

            metrics_storage.save_errors(
                timestamp=row['timestamp'],
                node=row['node'],
                absolute_error=absolute_error,
                relative_error=relative_error
            )

    except Exception as e:
        print(f"Error in update_and_predict: {str(e)}")


# Добавить планировщик
scheduler = BackgroundScheduler()
# scheduler.add_job(func=update_model_from_prometheus, trigger="interval", minutes=5)
scheduler.add_job(func=cluster_metrics.collect_cluster_metrics, trigger="interval", minutes=2)
scheduler.add_job(func=update_and_predict, trigger="interval", minutes=1)
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


@handle_exceptions
@app.route('/fit/<model_type>', methods=['POST'])
def fit_model(model_type):
    try:
        df = process_uploaded_file(request)

        if model_type == 'polynomial':
            if 'minutes_since_midnight' not in df.columns:
                df = add_time_features(df)
            models[model_type].fit(df[['minutes_since_midnight']], df[['minutes_since_midnight']])
        elif model_type == 'xgboost':
            models[model_type].partial_fit(df)
        elif model_type == 'sarima':
            models[model_type].update_data(df)
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        return jsonify({'status': f'{model_type} model updated'})

    except KeyError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/<model_type>', methods=['GET'])
@handle_exceptions
def predict(model_type):
    try:
        timestamp_str = request.args.get('timestamp')
        if not timestamp_str:
            return jsonify({'error': 'Missing timestamp'}), 400

        timestamp = pd.to_datetime(timestamp_str)
        result = models[model_type].predict(timestamp)

        # Сбор и сохранение предсказанных данных
        cluster_state = cluster_metrics.get_cluster_state()
        for node, metrics in cluster_state['nodes'].items():
            metrics_storage.save_metrics(
                timestamp=timestamp,
                node=node,
                cpu_actual=None,  # Предсказание, реальных данных пока нет
                cpu_predicted=result['cpu']
            )

        return jsonify({
            'model': model_type,
            'timestamp': timestamp.isoformat(),
            **result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/forecast', methods=['POST'])
def forecast():
    # Check if the request contains JSON data
    if request.is_json:
        data = request.get_json()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        # Read CSV file from the request
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        # Convert file to DataFrame
        df = pd.read_csv(StringIO(file.read().decode('utf-8')))

    # Validate DataFrame format and presence of timestamp
    if 'timestamp' not in df.columns:
        return jsonify({'error': 'CSV/JSON must contain timestamp column'}), 400

    # Convert timestamps to DateTime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # Predict for the next 15 minutes
    last_timestamp = df.index[-1]
    next_timestamp = last_timestamp + timedelta(minutes=15)

    prediction = {}

    # SARIMA models for each type of data
    if 'cpu' in df.columns:
        cpu_model = SARIMAX(df['cpu'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        cpu_results = cpu_model.fit()
        prediction['cpu'] = cpu_results.forecast(steps=1)[0]
    if 'memory' in df.columns:
        memory_model = SARIMAX(df['memory'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        memory_results = memory_model.fit()
        prediction['memory'] = memory_results.forecast(steps=1)[0]

    if 'requests' in df.columns:
        requests_model = SARIMAX(df['requests'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        requests_results = requests_model.fit()
        prediction['requests'] = requests_results.forecast(steps=1)[0]

    prediction['timestamp'] = next_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    return jsonify(prediction)


@app.route('/metrics/<model_type>')
def metrics(model_type):
    try:
        # Получаем прогноз от выбранной модели
        prediction = models[model_type].predict(pd.Timestamp.now())

        # Получаем максимальные значения из конфига
        max_cpu = float(config.get('DEFAULT', 'MaxCpu'))
        max_mem = float(config.get('DEFAULT', 'MaxMem'))

        # Вычисляем комбинированную метрику
        combined_metric = max(
            prediction['cpu'] / max_cpu * 100,
            prediction['memory'] / max_mem * 100
        )

        return jsonify({
            'value': round(combined_metric, 2),
            'timestamp': pd.Timestamp.now().isoformat()
        })

    except KeyError:
        return jsonify({'error': 'Invalid model type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics/<model_type>/cpu')
def metrics_cpu(model_type):
    try:
        prediction = models[model_type].predict(pd.Timestamp.now())
        max_cpu = float(config.get('DEFAULT', 'MaxCpu'))
        cpu_metric = prediction['cpu'] / max_cpu * 100

        return jsonify({
            'value': round(cpu_metric, 2),
            'timestamp': pd.Timestamp.now().isoformat()
        })
    except KeyError:
        return jsonify({'error': 'Invalid model type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics/<model_type>/memory')
def metrics_memory(model_type):
    try:
        prediction = models[model_type].predict(pd.Timestamp.now())
        max_mem = float(config.get('DEFAULT', 'MaxMem'))
        mem_metric = prediction['memory'] / max_mem * 100

        return jsonify({
            'value': round(mem_metric, 2),
            'timestamp': pd.Timestamp.now().isoformat()
        })
    except KeyError:
        return jsonify({'error': 'Invalid model type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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


# Добавьте новый endpoint для получения сравнения
@app.route('/metrics/comparison', methods=['GET'])
def get_comparison():
    node = request.args.get('node')
    comparison = metrics_comparator.get_comparison(node)
    return jsonify(comparison.to_dict(orient='records'))


@app.route('/metrics/errors', methods=['GET'])
def get_errors():
    try:
        start_date = pd.to_datetime(request.args.get('start_date'))
        end_date = pd.to_datetime(request.args.get('end_date'))

        errors = metrics_storage.get_errors(start_date, end_date)
        return jsonify(errors.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Добавляем новый endpoint для анализа точности
@app.route('/metrics/accuracy', methods=['GET'])
def get_accuracy():
    try:
        start_date = pd.to_datetime(request.args.get('start_date'))
        end_date = pd.to_datetime(request.args.get('end_date'))

        accuracy_data = metrics_storage.calculate_accuracy_over_time(start_date, end_date)
        return jsonify(accuracy_data.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Endpoint для получения последнего предсказания
@app.route('/predict/latest', methods=['GET'])
def get_latest_prediction():
    try:
        if not os.path.exists(PREDICTION_FILE):
            return jsonify({'error': 'No predictions available'}), 404

        prediction = pd.read_parquet(PREDICTION_FILE).iloc[-1].to_dict()
        return jsonify({
            'timestamp': prediction['timestamp'].isoformat(),
            'cpu': prediction['cpu'],
            'memory': prediction['memory']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
