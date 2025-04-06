from datetime import datetime, timedelta
import asyncio
import pandas as pd
from models.xgboost import XGBoostModel

class MetricsClient:
    def __init__(self, metrics_collector: MetricsCollector, xgboost_model: XGBoostModel, uuid: str):
        self.metrics_collector = metrics_collector
        self.xgboost_model = xgboost_model
        self.uuid = uuid

    async def _update_model(self):
        try:
            # Получаем данные за последнюю минуту
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=1)
            
            # Получаем метрики CPU
            cpu_data = await self.metrics_collector.get_cpu_metrics(self.uuid, start_time, end_time)
            
            if not cpu_data.empty:
                # Преобразуем данные для модели
                train_data = pd.DataFrame({
                    'timestamp': cpu_data['timestamp'],
                    'cpu': cpu_data['value']
                })
                
                # Дообучаем модель
                await self.xgboost_model.partial_fit(train_data)
                print("Model updated with new data")
            
        except Exception as e:
            print(f"Error updating model: {e}")

    async def start(self, interval: int = 60):
        while True:
            await self._update_model()
            await asyncio.sleep(interval)
