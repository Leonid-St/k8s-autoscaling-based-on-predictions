import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.xgboost import XGBoostModel
from utils.cluster_metrics import NodeMetrics

@pytest.fixture
def cluster_metrics():
    return NodeMetrics("http://localhost:9090")

@pytest.fixture
def xgboost_model(cluster_metrics):
    return XGBoostModel(cluster_metrics)

def generate_synthetic_data(start_time, num_points=100):
    """Генерация синтетических данных для тестирования"""
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_points)]
    cpu = np.sin(np.linspace(0, 4 * np.pi, num_points)) * 0.5 + 0.5
    memory = np.cos(np.linspace(0, 4 * np.pi, num_points)) * 0.5 + 0.5
    return pd.DataFrame({
        'timestamp': timestamps,
        'cpu': cpu,
        'memory': memory
    })

def test_xgboost_training_process(xgboost_model, tmp_path):
    # Генерация синтетических данных
    start_time = datetime.now()
    synthetic_data = generate_synthetic_data(start_time, num_points=5000)  # Увеличиваем количество данных
    
    # Файл для записи результатов
    results_file = tmp_path / "training_results.csv"
    results = []

    # Постепенное обучение модели
    for i in range(0, len(synthetic_data), 200):  # Увеличиваем размер батча для обучения
        # Берем часть данных для обучения
        train_data = synthetic_data.iloc[i:i+200]
        
        # Обучаем модель
        xgboost_model.partial_fit(train_data)
        
        # Делаем предсказания для всех данных
        predictions = []
        for timestamp in synthetic_data['timestamp']:
            pred = xgboost_model.predict(timestamp)
            predictions.append(pred)
        
        # Сохраняем результаты
        for j, (true_values, pred_values) in enumerate(zip(synthetic_data.to_dict('records'), predictions)):
            results.append({
                'timestamp': true_values['timestamp'],
                'cpu_actual': true_values['cpu'],
                'memory_actual': true_values['memory'],
                'cpu_predicted': pred_values['cpu'],
                'memory_predicted': pred_values['memory'],
                'training_step': i // 200
            })
    
    # Сохраняем результаты в CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    
    # Проверяем, что файл создан и содержит данные
    assert results_file.exists()
    assert len(results_df) > 0
    
    # Проверяем, что ошибка уменьшается с течением времени
    results_df['cpu_error'] = abs(results_df['cpu_actual'] - results_df['cpu_predicted'])
    results_df['memory_error'] = abs(results_df['memory_actual'] - results_df['memory_predicted'])
    
    # Группируем по шагам обучения и проверяем, что ошибка уменьшается в целом
    grouped = results_df.groupby('training_step')[['cpu_error', 'memory_error']].mean()
    
    # Проверяем, что ошибка уменьшилась хотя бы на 0.1% в конце обучения
    cpu_error_reduction = (grouped['cpu_error'].iloc[0] - grouped['cpu_error'].iloc[-1]) / grouped['cpu_error'].iloc[0]
    memory_error_reduction = (grouped['memory_error'].iloc[0] - grouped['memory_error'].iloc[-1]) / grouped['memory_error'].iloc[0]
    
    # Проверяем, что ошибка уменьшилась хотя бы на 0.1% в конце обучения
    assert cpu_error_reduction > 0.001, f"CPU error reduction was {cpu_error_reduction:.4f}, expected > 0.001"
    assert memory_error_reduction > 0.001, f"Memory error reduction was {memory_error_reduction:.4f}, expected > 0.001"

def test_partial_fit(xgboost_model):
    data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'cpu': [0.5],
        'memory': [0.3]
    })
    xgboost_model.partial_fit(data)
    assert xgboost_model.cpu_model is not None
    assert xgboost_model.memory_model is not None

def test_predict(xgboost_model):
    # Обучаем модель перед предсказанием
    data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'cpu': [0.5],
        'memory': [0.3]
    })
    xgboost_model.partial_fit(data)
    
    # Теперь можно предсказать
    timestamp = datetime.now()
    prediction = xgboost_model.predict(timestamp)
    assert 'cpu' in prediction
    assert 'memory' in prediction 
