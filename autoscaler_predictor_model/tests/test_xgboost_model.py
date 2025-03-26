import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.xgboost import XGBoostModel
from utils.cluster_metrics import NodeMetrics
from data_synthesizer import synthesize_data
from pathlib import Path

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

def test_xgboost_training_process(xgboost_model):
    # Создаем папку для артефактов тестов
    artifacts_dir = Path('test_artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    
    # Генерация синтетических данных
    start_time = datetime.now()
    synthetic_data = synthesize_data(num_weeks=4, type='resource')  # Увеличили до 4 недель
    
    # Преобразуем индекс timestamp в колонку
    synthetic_data = synthetic_data.reset_index()
    
    # Файл для записи результатов
    results_file = artifacts_dir / "xgboost_training_results.csv"
    results = []

    # Постепенное обучение модели
    for i in range(0, len(synthetic_data), 50):  # Уменьшили размер батча до 50
        # Берем часть данных для обучения
        train_data = synthetic_data.iloc[i:i+50]
        
        # Обучаем только CPU модель
        if 'cpu' in train_data.columns:
            # Убираем нормализацию
            cpu_target = train_data[['cpu']].values
            
            if xgboost_model._cpu_fitted:
                xgboost_model.cpu_model.fit(
                    xgboost_model._create_features(train_data), 
                    cpu_target,
                    xgb_model=xgboost_model.cpu_model.get_booster()
                )
            else:
                xgboost_model.cpu_model.fit(
                    xgboost_model._create_features(train_data), 
                    cpu_target
                )
                xgboost_model._cpu_fitted = True
        
        # Делаем предсказания только для CPU
        predictions = []
        for timestamp in synthetic_data['timestamp']:
            features = xgboost_model._create_features(pd.DataFrame({'timestamp': [timestamp]}))
            cpu_pred = xgboost_model.cpu_model.predict(features)
            predictions.append({'cpu': cpu_pred[0]})
        
        # Сохраняем результаты
        for j, (true_values, pred_values) in enumerate(zip(synthetic_data.to_dict('records'), predictions)):
            results.append({
                'timestamp': true_values['timestamp'],
                'cpu_actual': true_values['cpu'],
                'cpu_predicted': pred_values['cpu'],
                'training_step': i // 50
            })
    
    # Сохраняем результаты в CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    
    # Проверяем, что файл создан и содержит данные
    assert results_file.exists()
    assert len(results_df) > 0
    
    # Проверяем, что ошибка уменьшается с течением времени
    results_df['cpu_error'] = abs(results_df['cpu_actual'] - results_df['cpu_predicted'])
    
    # Группируем по шагам обучения и проверяем, что ошибка уменьшается в целом
    grouped = results_df.groupby('training_step')['cpu_error'].mean()
    
    # Проверяем, что ошибка уменьшилась хотя бы на 1% в конце обучения
    cpu_error_reduction = (grouped.iloc[0] - grouped.iloc[-1]) / grouped.iloc[0]
    assert cpu_error_reduction > 0.01, f"CPU error reduction was {cpu_error_reduction:.4f}, expected > 0.01"

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
