from datetime import datetime, timedelta
import pandas as pd
from typing import List, Callable, Awaitable
import logging
from models.xgboost_model import XGBoostModel
from storage.storage_service import StorageService

logger = logging.getLogger(__name__)

class PredictorService:
    def __init__(self, model: XGBoostModel, storage: StorageService, node_id: str):
        self.model = model
        self.storage = storage
        self.node_id = node_id
        self._observers: List[Callable[[], Awaitable[None]]] = []
        self.update_count = 0

    def add_observer(self, observer: Callable[[], Awaitable[None]]):
        self._observers.append(observer)

    async def _notify_observers(self):
        for observer in self._observers:
            await observer()

    async def on_model_updated(self):
        self.update_count += 1
        if self.update_count >= 5:
            await self._make_predictions()
            self.update_count = 0

    async def _make_predictions(self):
        try:
            # Получаем текущее время
            now = datetime.now()
            
            # Генерируем временные метки для предсказаний (5 минут вперед)
            prediction_times = [now + timedelta(minutes=i) for i in range(1, 6)]
            
            # Выполняем предсказания
            predictions = []
            for timestamp in prediction_times:
                prediction = self.model.predict(timestamp)
                predictions.append({
                    'timestamp': timestamp,
                    'cpu': prediction['cpu'],
                    'memory': prediction.get('memory', 0)
                })
            
            # Сохраняем предсказания в хранилище
            for prediction in predictions:
                await self.storage.save_prediction(
                    timestamp=prediction['timestamp'],
                    node=self.node_id,
                    model_type='xgboost',
                    prediction=prediction
                )
            
            logger.info(f"Predictions saved for node {self.node_id}")
            await self._notify_observers()
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}") 
