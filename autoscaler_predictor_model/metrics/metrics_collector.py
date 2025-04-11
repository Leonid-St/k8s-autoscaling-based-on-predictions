from metrics.metrics_fetcher import MetricsFetcher
from storage.storage_service import StorageService
import logging
from typing import List, Callable, Awaitable

logger = logging.getLogger("uvicorn")


class MetricsCollector:
    def __init__(self, *, uuid: str, fetcher: MetricsFetcher, storage: StorageService):
        self.uuid = uuid
        self.fetcher = fetcher
        self.storage = storage
        self._observers: List[Callable[[], Awaitable[None]]] = []

    def add_observer(self, observer: Callable[[], Awaitable[None]]):
        self._observers.append(observer)

    async def _notify_observers(self):
        for observer in self._observers:
            await observer()

    async def collect(self):
        metric = await self.fetcher.get_cpu_memory_metrics_1m(self.uuid)
        await self.storage.save_actual(metrics=metric, node=self.uuid)
        await self._notify_observers()
