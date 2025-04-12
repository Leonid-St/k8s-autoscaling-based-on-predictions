from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from storage.storage_service import StorageService
import logging

logger = logging.getLogger("uvicorn.error")


class ComparatorService:
    def __init__(self, storage: StorageService, node_id: str):
        self.storage = storage
        self.node_id = node_id

    async def compare_and_save_errors(self):
        try:
            # Получаем текущее время
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)

            # Получаем фактические данные за последние 5 минут
            actual_data = await self.storage.get_actual_range(start_time, end_time, self.node_id)

            # Получаем предсказания за последние 5 минут
            predicted_data = await self.storage.get_predictions_range(start_time, end_time, self.node_id)

            if not actual_data.empty and not predicted_data.empty:
                # Объединяем данные по временным меткам
                merged_data = pd.merge(actual_data, predicted_data, on='timestamp', suffixes=('_actual', '_predicted'))

                # Вычисляем ошибки
                rmse = np.sqrt(((merged_data['cpu_actual'] - merged_data['cpu_predicted']) ** 2).mean())
                mae = np.abs(merged_data['cpu_actual'] - merged_data['cpu_predicted']).mean()

                # Сохраняем ошибки в таблицу
                await self.storage.save_error(
                    start_time=start_time,
                    end_time=end_time,
                    node=self.node_id,
                    model_type='xgboost',
                    error_metrics={'rmse': rmse, 'mae': mae}
                )
                logger.info(f"Errors saved for node {self.node_id} from {start_time} to {end_time}")
            else:
                logger.warning(
                    f"No actual or predicted data found for node {self.node_id} from {start_time} to {end_time}")

        except Exception as e:
            logger.error(f"Error comparing and saving errors: {e}")
