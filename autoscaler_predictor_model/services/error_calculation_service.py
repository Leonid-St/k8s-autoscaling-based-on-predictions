from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from storage.storage_service import StorageService

class ErrorCalculationService:
    def __init__(self, storage: StorageService):
        self.storage = storage

    def calculate_errors(self, start_date: datetime, end_date: datetime, node: str, model_type: str):
        actuals = self.storage.get_actuals(start_date, end_date, node)
        predictions = self.storage.get_predictions(start_date, end_date, node, model_type)
        
        if actuals.empty or predictions.empty:
            return None

        merged = pd.merge(actuals, predictions, on=['timestamp', 'node'], suffixes=('_actual', '_predicted'))
        
        error_metrics = {
            'mse': mean_squared_error(merged['cpu_actual'], merged['cpu_predicted']),
            'mae': mean_absolute_error(merged['cpu_actual'], merged['cpu_predicted'])
        }
        
        self.storage.save_error(datetime.now(), node, model_type, error_metrics)
        return error_metrics 
