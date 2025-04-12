from datetime import datetime, timedelta
import pandas as pd
from typing import List, Callable, Awaitable
import logging

from models.metric_data import MetricData
from models.xgboost import XGBoostModel
from storage.storage_service import StorageService

logger = logging.getLogger("uvicorn.error")


class PredictorService:
    def __init__(self, model: XGBoostModel, storage: StorageService, node_id: str):
        self.model = model
        self.storage = storage
        self.node_id = node_id
        self._observers: List[Callable[[MetricData], Awaitable[None]]] = []

    def add_observer(self, observer: Callable[[MetricData], Awaitable[None]]):
        self._observers.append(observer)

    async def _notify_observers(self, metric_data: MetricData):
        for observer in self._observers:
            await observer(metric_data)

    async def on_model_updated(self, metric_data: MetricData):
        await self._make_predictions(metric_data)

    async def _make_predictions(self, metric_data: MetricData):
        try:
            prediction_time = metric_data.timestamp + timedelta(minutes=1)
            prediction = self.model.predict(prediction_time)

            await self.storage.save_prediction(
                timestamp=prediction_time,
                node=self.node_id,
                prediction={
                    'timestamp': prediction_time,
                    'cpu': prediction['cpu'],
                    'memory': prediction.get('memory', 0)
                }
            )

            await self._notify_observers(metric_data)
            logger.info(f"Prediction saved for node {self.node_id}")
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
