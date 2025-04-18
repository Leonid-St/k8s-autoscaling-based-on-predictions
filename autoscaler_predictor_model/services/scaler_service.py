import asyncio
import requests
from datetime import datetime
import logging

from config import EnvConfig
from storage.storage_service import StorageService
from models.metric_data import MetricData

logger = logging.getLogger("uvicorn.error")


class ScalerService:
    def __init__(self, storage: StorageService, node_id: str, keda_scaler_address: str):
        self.storage = storage
        self.node_id = node_id
        self.keda_scaler_address = keda_scaler_address

    async def scale(self):
        """
        Периодически проверяет последнее предсказание и отправляет метрику в Keda при необходимости.
        """
        latest_prediction = await self.storage.get_latest_prediction(
            node=self.node_id,
        )

        if latest_prediction:
            cpu_prediction = latest_prediction.metrics.get('cpu')
            if cpu_prediction is not None and cpu_prediction >= 100:  # Условие масштабирования
                await self._send_metrics_to_keda(cpu_prediction)
        else:
            logger.info("Предсказания не найдены.")

    async def _send_metrics_to_keda(self, cpu_prediction: float):
        """
        Отправляет метрику CPU в Keda через external-push.
        """
        metrics_url = f"{self.keda_scaler_address}/push"
        metrics_payload = {"metricName": "cpu", "metricValue": cpu_prediction}  # Имя метрики "cpu" и значение
        try:
            response = requests.post(metrics_url, json=metrics_payload)
            response.raise_for_status()  # Проверка на ошибки HTTP
            logger.info(f"Метрика отправлена в Keda: CPU = {cpu_prediction}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка отправки метрики в Keda: {e}") 
