from datetime import datetime
from storage.storage_service import StorageService
from models.xgboost import XGBoostModel

class PredictionService:
    def __init__(self, model: XGBoostModel, storage: StorageService):
        self.model = model
        self.storage = storage

    def make_prediction(self, timestamp: datetime, node: str, model_type: str = 'xgboost'):
        prediction = self.model.predict(timestamp)
        self.storage.save_prediction(timestamp, node, model_type, prediction)
        return prediction 
