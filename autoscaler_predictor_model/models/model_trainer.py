import pandas as pd
from datetime import datetime, timedelta
from models.xgboost import XGBoostModel

class ModelTrainer:
    def __init__(self, metrics_storage, xgboost_model):
        self.metrics_storage = metrics_storage
        self.xgboost_model = xgboost_model
        self.metrics_storage.add_observer(self)  # Регистрируем себя как наблюдателя

    def update(self, file_path):
        # Обрабатываем новый файл
        latest_metrics = pd.read_csv(file_path)
        if not latest_metrics.empty:
            self.xgboost_model.fit(latest_metrics)
            print("Model updated with new data") 
