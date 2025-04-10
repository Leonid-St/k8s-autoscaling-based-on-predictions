from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

class StorageService(ABC):
    @abstractmethod
    def save_prediction(self, timestamp: datetime, node: str, model_type: str, prediction_type: str, prediction: dict):
        pass

    @abstractmethod
    async def save_actual(self,*, node: str, metrics: dict):
        pass

    @abstractmethod
    def save_error(self, timestamp: datetime, node: str, model_type: str, error_metrics: dict):
        pass

    @abstractmethod
    def get_predictions(self, start_date: datetime, end_date: datetime, node: str = None, model_type: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_actuals(self, start_date: datetime, end_date: datetime, node: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_errors(self, start_date: datetime, end_date: datetime, node: str = None, model_type: str = None) -> pd.DataFrame:
        pass 
