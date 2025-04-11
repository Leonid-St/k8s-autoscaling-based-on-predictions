from datetime import datetime, timedelta
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from services.storage_service import StorageService
from models.xgboost_model import XGBoostModel
import logging
from typing import List, Callable, Awaitable

logger = logging.getLogger(__name__)

class TeacherService:
    def __init__(self, model: XGBoostModel, storage: StorageService, node_id: str, scheduler: AsyncIOScheduler):
        self.model = model
        self.storage = storage
        self.node_id = node_id
        self.scheduler = scheduler
        self._observers: List[Callable[[], Awaitable[None]]] = []

    def add_observer(self, observer: Callable[[], Awaitable[None]]):
        self._observers.append(observer)

    async def _notify_observers(self):
        for observer in self._observers:
            await observer()

    async def on_new_data(self):
        try:
            actual_data = await self.storage.get_actual(self.node_id)
            
            if actual_data:
                train_data = pd.DataFrame({
                    'timestamp': [actual_data['timestamp']],
                    'cpu': [actual_data['cpu']],
                    'memory': [actual_data['memory']]
                })
                
                self.model.fit(train_data)
                logger.info(f"Model trained with new data for node {self.node_id}")
                await self._notify_observers()
            
        except Exception as e:
            logger.error(f"Error training model: {e}")

    def start(self, interval: int = 60):
        # Добавляем задачу в уже запущенный планировщик
        self.scheduler.add_job(
            self.on_new_data,
            'interval',
            seconds=interval,
            next_run_time=datetime.now()
        )
        # Не вызываем scheduler.start(), так как он уже запущен в main.py 
