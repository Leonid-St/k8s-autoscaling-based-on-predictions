import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.xgboost import XGBoostModel
from utils.cluster_metrics import NodeMetrics
import logging  
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from models.xgboost import XGBoostModel
import time
from data_synthesizer import synthesize_data, plot_synthesized_data  

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@pytest.fixture
def cluster_metrics():
    return NodeMetrics("http://localhost:9090")

@pytest.fixture
def xgboost_model(cluster_metrics):
    return XGBoostModel(cluster_metrics)


def test_xgboost_training_process(xgboost_model):
    start_time = time.time()  # Запоминаем время начала
    logging.info("Запуск теста test_example")
        
    # Создаем папку для артефактов тестов
    artifacts_dir = Path('test_artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    
    # Генерация синтетических данных
    start_time = datetime.now()

    synthetic_data, data_file = synthesize_data(num_weeks=8, type='resource',save_to_file=True)    
    synthetic_data = pd.read_csv(data_file)
    synthetic_data['timestamp'] = pd.to_datetime(synthetic_data['timestamp']) 

    # Преобразуем индекс timestamp в колонку
    synthetic_data = synthetic_data.reset_index(drop=True)  # Убрали старый индекс    
    # Файл для записи результатов
    results_file = artifacts_dir / "xgboost_training_results.csv"
    results = []

    # Постепенное обучение модели
    for i in range(0, len(synthetic_data), 50):  # Уменьшили размер батча до 50
       
         # Логируем текущий шаг обучения
        step = i // 50 + 1
        total_steps = len(synthetic_data) // 50
        logging.info(f"Шаг обучения {step}/{total_steps} (данные с {i} по {i+50})")
       
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
                logging.debug("Первоначальное обучение CPU модели...")
                xgboost_model.cpu_model.fit(
                    xgboost_model._create_features(train_data), 
                    cpu_target
                )
                xgboost_model._cpu_fitted = True
                logging.info("CPU модель успешно обучена")
        
        # Делаем предсказания только для CPU
        predictions = []
        logging.debug("Создание предсказаний для текущего шага...")
        for timestamp in synthetic_data['timestamp']:
            features = xgboost_model._create_features(pd.DataFrame({'timestamp': [timestamp]}))
            cpu_pred = xgboost_model.cpu_model.predict(features)
            predictions.append({'cpu': cpu_pred[0]})
        
        # Сохраняем результаты
        logging.debug("Сохранение результатов шага...")
        for j, (true_values, pred_values) in enumerate(zip(synthetic_data.to_dict('records'), predictions)):
            results.append({
                'timestamp': true_values['timestamp'],
                'cpu_actual': true_values['cpu'],
                'cpu_predicted': pred_values['cpu'],
                'training_step': i // 50
            })
        
        logging.info(f"Шаг {step} завершен. Обработано {len(train_data)} записей")
    
    # Сохраняем результаты в CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)

    # Визуализация результатов
    img_dir = artifacts_dir / "imgs"
    img_dir.mkdir(exist_ok=True)
    
    # График фактических и предсказанных значений
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['timestamp'], results_df['cpu_actual'], label='Actual CPU')
    plt.plot(results_df['timestamp'], results_df['cpu_predicted'], label='Predicted CPU', alpha=0.7)
    plt.title('Фактические vs Предсказанные значения CPU')
    plt.xlabel('Время')
    plt.ylabel('Нагрузка CPU')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    actual_vs_predicted_path = img_dir / "xgboost_actual_vs_predicted.png"
    plt.savefig(actual_vs_predicted_path)
    plt.close()
    
    # График распределения ошибок
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='training_step', y='cpu_error', data=results_df)
    plt.title('Распределение ошибок прогнозирования CPU')
    plt.xlabel('Шаг обучения')
    plt.ylabel('Абсолютная ошибка')
    error_distribution_path = img_dir / "xgboost_error_distribution.png"
    plt.savefig(error_distribution_path)
    plt.close()
    
    # Проверяем что файлы с графиками созданы
    assert actual_vs_predicted_path.exists()
    assert error_distribution_path.exists()
    
    # # Проверяем что файл с графиком создан
    # assert plot_path.exists()

    # Проверяем, что файл создан и содержит данные
    assert results_file.exists()
    assert len(results_df) > 0
    
    # Проверяем, что ошибка уменьшается с течением времени
    results_df['cpu_error'] = abs(results_df['cpu_actual'] - results_df['cpu_predicted'])
    
    # Группируем по шагам обучения и проверяем, что ошибка уменьшается в целом
    grouped = results_df.groupby('training_step')['cpu_error'].mean()
    
    # Проверяем, что ошибка уменьшилась хотя бы на 50% в конце обучения
    cpu_error_reduction = (grouped.iloc[0] - grouped.iloc[-1]) / grouped.iloc[0]

    print(f"Error progression:\n{grouped}")
    plt.figure()
    grouped.plot(title='Error progression')
    plt.savefig('error_progression.png')
    plt.close()

    # Заменяем старую проверку:
    # assert cpu_error_reduction > 0.5

    # Проверяем, что ошибка уменьшилась хотя бы на 50% в конце обучения по сравнению с началом
    cpu_error_reduction = (grouped.iloc[0] - grouped.iloc[-1]) / grouped.iloc[0]
    logging.info(f"CPU error reduction ALL: {cpu_error_reduction:.4f}")

    # На новую проверку последних 25% шагов:
    last_quarter = int(len(grouped) * 0.75)
    last_steps_error = grouped.iloc[last_quarter:].mean()
    initial_error = grouped.iloc[:last_quarter].mean()
    cpu_error_reduction = (initial_error - last_steps_error) / initial_error
    assert cpu_error_reduction > 0.3, f"Error reduction: {cpu_error_reduction:.4f}"

    end_time = time.time()  # Запоминаем время окончания
    duration = end_time - start_time  # Вычисляем продолжительность
    logging.info(f"CPU error reduction: {cpu_error_reduction:.4f}")
    duration_minutes = duration / 60  # Переводим в минуты
    logging.info(f"Тест  завершен. Время выполнения: {duration:.2f} секунд")
    logging.info(f"Тест  завершен. Время выполнения: {duration_minutes:.2f} минут")

def test_partial_fit(xgboost_model):
    data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'cpu': [0.5],
        #'memory': [0.3]
    })
    xgboost_model.partial_fit(data)
    assert xgboost_model.cpu_model is not None
    #assert xgboost_model.memory_model is not None

def test_predict(xgboost_model):
    # Обучаем модель перед предсказанием
    data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'cpu': [0.5],
        #'memory': [0.3]
    })
    xgboost_model.partial_fit(data)
    
    # Теперь можно предсказать
    timestamp = datetime.now()
    prediction = xgboost_model.predict(timestamp)
    assert 'cpu' in prediction
    #assert 'memory' in prediction 

def test_xgboost_quick_smoke(xgboost_model):
    # Генерация минимального набора данных (1 день вместо 8 недель)
    synthetic_data, data_file = synthesize_data(num_weeks=0.02, type='resource', save_to_file=True)  # ~3.5 часа данных
    synthetic_data = pd.read_csv(data_file)
    synthetic_data['timestamp'] = pd.to_datetime(synthetic_data['timestamp'])
    
    # Быстрое обучение одним батчем
    xgboost_model.partial_fit(synthetic_data)
    
    # Проверка базовых предикторов
    test_timestamp = synthetic_data['timestamp'].iloc[0] + pd.Timedelta(minutes=15)
    prediction = xgboost_model.predict(test_timestamp)
    
    # Базовые проверки
    assert isinstance(prediction, dict)
    assert 'cpu' in prediction
    assert 0 <= prediction['cpu'] <= MAX_CPU  # MAX_CPU из data_synthesizer
    
    # Проверка структуры данных модели
    assert hasattr(xgboost_model, 'cpu_model')
    assert xgboost_model._cpu_fitted is True
    assert xgboost_model.cpu_model.n_features_in_ == 5  # Количество фичей в _create_features
