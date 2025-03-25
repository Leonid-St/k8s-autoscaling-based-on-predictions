from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import configparser
from apscheduler.schedulers.background import BackgroundScheduler
from prometheus_client import PrometheusDataCollector

from utils.file_processor import process_uploaded_file
from models.polynomial import PolynomialModel
from models.xgboost import XGBoostModel
from models.sarima import SARIMAModel
from utils.cluster_metrics import NodeMetrics

MODEL_PATH = '/app/models/'

app = Flask(__name__)

# Global variable to store the model
cpu_model = None
memory_model = None
requests_model = None

# Добавить глобальные переменные
historical_data = pd.DataFrame()
DATA_RETENTION = timedelta(hours=4)  # Храним 4 часа данных

config = configparser.ConfigParser()
config.read('config.ini')

# CPU: container_cpu_usage_seconds_total
# Memory: container_memory_working_set_bytes
# Для автоскейлинга приложений: http_requests_total

# Добавить в конфиг
PROMETHEUS_URL = config.get('PROMETHEUS', 'Url')
CPU_QUERY = config.get('PROMETHEUS', 'CpuQuery')
MEMORY_QUERY = config.get('PROMETHEUS', 'MemoryQuery')

# Инициализация моделей
models = {
    'polynomial': PolynomialModel(),
    'xgboost': XGBoostModel(),
    'sarima': SARIMAModel(data_retention='4H')
}

# Инициализация коллектора
prom_collector = PrometheusDataCollector(PROMETHEUS_URL, CPU_QUERY, MEMORY_QUERY)

# Инициализация метрик кластера
cluster_metrics = NodeMetrics(PROMETHEUS_URL)


def update_model_from_prometheus():
    try:
        new_data = prom_collector.get_new_data()
        if not new_data.empty:
            models['xgboost'].partial_fit(new_data)
            print(f"Model updated with {len(new_data)} new samples")
    except Exception as e:
        print(f"Error updating model: {str(e)}")


# Добавить планировщик
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_model_from_prometheus, trigger="interval", minutes=5)
scheduler.add_job(func=cluster_metrics.collect_cluster_metrics, trigger="interval", minutes=2)
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

        if model_type == 'polynomial':
            features = [[timestamp.hour * 60 + timestamp.minute]]
            result = models[model_type].predict(features, 'resource')
        elif model_type == 'xgboost':
            result = models[model_type].predict(timestamp)
        elif model_type == 'sarima':
            result = models[model_type].forecast()
            if result is None:
                return jsonify({'error': 'Not enough historical data for SARIMA'}), 400
        else:
            return jsonify({'error': 'Invalid model type'}), 400

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
    if 'cpu' in df.columns and 'memory' in df.columns:
        cpu_model = SARIMAX(df['cpu'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        cpu_results = cpu_model.fit()
        prediction['cpu'] = cpu_results.forecast(steps=1)[0]

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


def scaling_decision(prediction):
    config = {
        'scale_up': float(config.get('AUTOSCALER', 'ScaleUpThreshold')),
        'scale_down': float(config.get('AUTOSCALER', 'ScaleDownThreshold')),
        'max_nodes': int(config.get('AUTOSCALER', 'MaxNodes')),
        'min_nodes': int(config.get('AUTOSCALER', 'MinNodes')),
        'step': int(config.get('AUTOSCALER', 'ScaleStep'))
    }

    current_state = cluster_metrics.get_cluster_state()
    predicted_load = prediction['value']

    if predicted_load > config['scale_up'] and current_state['total_nodes'] < config['max_nodes']:
        return {'action': 'scale_up', 'by': config['step']}
    elif predicted_load < config['scale_down'] and current_state['total_nodes'] > config['min_nodes']:
        return {'action': 'scale_down', 'by': config['step']}
    return {'action': 'no_op'}
