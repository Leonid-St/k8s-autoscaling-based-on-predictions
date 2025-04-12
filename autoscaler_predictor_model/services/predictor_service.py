from datetime import datetime, timedelta
import pandas as pd
from typing import List, Callable, Awaitable
import logging
from models.xgboost import XGBoostModel
from storage.storage_service import StorageService

logger = logging.getLogger("uvicorn.error")


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
        await self._make_predictions()

    async def _make_predictions(self):
        try:
            now = datetime.now()
            prediction_time = now + timedelta(minutes=1)

            prediction = self.model.predict(prediction_time)

            await self.storage.save_prediction(
                timestamp=prediction_time,
                node=self.node_id,
                model_type='xgboost',
                prediction={
                    'timestamp': prediction_time,
                    'cpu': prediction['cpu'],
                    'memory': prediction.get('memory', 0)
                }
            )
            logger.info(f"Prediction saved for node {self.node_id}")
            await self._notify_observers()
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
