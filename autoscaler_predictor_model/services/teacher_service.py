from datetime import datetime, timedelta
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from models.metric_data import MetricData
from storage.storage_service import StorageService
from models.xgboost import XGBoostModel
import logging
from typing import List, Callable, Awaitable

logger = logging.getLogger("uvicorn.error")


class TeacherService:
    def __init__(self, model: XGBoostModel, storage: StorageService, node_id: str,
                 # scheduler: AsyncIOScheduler
                 ):
        self.model = model
        # if model._fitted is False:
        #
        #     model.fit()
        self.storage = storage
        self.node_id = node_id
        # self.scheduler = scheduler
        self._observers: List[Callable[[MetricData], Awaitable[None]]] = []

    def add_observer(self, observer: Callable[[MetricData], Awaitable[None]]):
        self._observers.append(observer)

    async def _notify_observers(self, metric_data: MetricData):
        for observer in self._observers:
            await observer(metric_data)

    async def on_new_data(self, metric_data: MetricData):
        try:
            train_data = pd.DataFrame({
                'timestamp': [metric_data.timestamp],
                'cpu': [metric_data.metrics['cpu']],
                'memory': [metric_data.metrics['memory']]
            })
            self.model.fit(train_data)
            logger.info(f"Model trained with new data for node {self.node_id}")
            await self._notify_observers(metric_data)
        except Exception as e:
            logger.error(f"Error training model: {e}")

    # def start(self, interval: int = 60):
    #     # Добавляем задачу в уже запущенный планировщик
    #     self.scheduler.add_job(
    #         self.on_new_data,
    #         'interval',
    #         seconds=interval,
    #         next_run_time=datetime.now()
    #     )
    # Не вызываем scheduler.start(), так как он уже запущен в main.py
