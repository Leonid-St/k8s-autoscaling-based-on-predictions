from datetime import datetime, timedelta
from typing import List

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.metric_data import MetricData
from storage.storage_service import StorageService
import logging
from dataclasses import dataclass
from models.errors_metrics import ErrorsMetrics, ErrorsProba

logger = logging.getLogger("uvicorn.error")


class ComparatorService:
    def __init__(self, storage: StorageService, node_id: str):
        self.storage = storage
        self.node_id = node_id
        self.actual_metrics: List[MetricData] = []
        self.predicted_metrics: List[MetricData] = []

    def calculate_errors(self, *, actual_metrics, predicted_metrics) -> ErrorsMetrics:
        # Извлекаем значения CPU и Memory из списков MetricData
        actual_cpu = [metric.metrics['cpu'] for metric in actual_metrics]
        actual_memory = [metric.metrics['memory'] for metric in actual_metrics]

        predicted_cpu = [metric.metrics['cpu'] for metric in predicted_metrics]
        predicted_memory = [metric.metrics['memory'] for metric in predicted_metrics]

        # Вычисляем MSE для CPU и Memory
        mse_cpu = mean_squared_error(actual_cpu, predicted_cpu)
        mse_memory = mean_squared_error(actual_memory, predicted_memory)

        # Вычисляем MAE для CPU и Memory
        mae_cpu = mean_absolute_error(actual_cpu, predicted_cpu)
        mae_memory = mean_absolute_error(actual_memory, predicted_memory)

        # Вычисляем RMSE для CPU и Memory
        rmse_cpu = np.sqrt(mse_cpu)
        rmse_memory = np.sqrt(mse_memory)

        cpu_metrics = ErrorsProba(mse=mse_cpu, mae=mae_cpu, rmse=rmse_cpu)
        memory_metrics = ErrorsProba(mse=mse_memory, mae=mae_memory, rmse=rmse_memory)

        return ErrorsMetrics(
            cpu=cpu_metrics,
            memory=memory_metrics,
            start_time=actual_metrics[0].timestamp,
            end_time=actual_metrics[-1].timestamp,
        )

    async def compare_and_save_errors(self, metric_data: MetricData):
        if len(self.actual_metrics) < 5:
            self.actual_metrics.append(metric_data)
            predicted_metric = await self.storage.get_prediction(timestamp=metric_data.timestamp, node=self.node_id)
            if predicted_metric is None:
                self.actual_metrics = []
                self.predicted_metrics = []
                logger.warning("predicted metric with timestamp:" + str(metric_data.timestamp) +
                               "not found. Metrics in comparator for 5 min are cleaning")
                return
            self.predicted_metrics.append(predicted_metric)
            return
        try:
            errors = self.calculate_errors(actual_metrics=self.actual_metrics, predicted_metrics=self.predicted_metrics)
            await self.storage.save_error(
                node=self.node_id,
                error_metrics=errors
            )
            logger.info(f"Errors saved for node {self.node_id} from {errors.start_time} to {errors.end_time}")
            self.actual_metrics = []
            self.predicted_metrics = []

        except Exception as e:
            logger.error(f"Error comparing and saving errors: {e}")
            self.actual_metrics = []
            self.predicted_metrics = []
