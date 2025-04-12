from metrics.metrics_fetcher import MetricsFetcher
from models.metric_data import MetricData
from storage.storage_service import StorageService
import logging
from typing import List, Callable, Awaitable

logger = logging.getLogger("uvicorn")


class MetricsCollector:
    def __init__(self, *, uuid: str, fetcher: MetricsFetcher, storage: StorageService):
        self.uuid = uuid
        self.fetcher = fetcher
        self.storage = storage
        self._observers: List[Callable[[MetricData], Awaitable[None]]] = []

    def add_observer(self, observer: Callable[[MetricData], Awaitable[None]]):
        self._observers.append(observer)

    async def _notify_observers(self, metric_data: MetricData):
        for observer in self._observers:
            await observer(metric_data)

    async def collect(self):
        metric = await self.fetcher.get_cpu_memory_metrics_1m(self.uuid)
        metric_data = MetricData(timestamp=metric['timestamp'], metrics=metric)
        await self.storage.save_actual(metrics=metric_data, node=self.uuid)
        await self._notify_observers(metric_data=metric_data)
