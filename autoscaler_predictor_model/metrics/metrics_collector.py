from metrics.metrics_fetcher import MetricsFetcher
from storage.storage_service import StorageService
import logging

logger = logging.getLogger("uvicorn")


class MetricsCollector:
    def __init__(self, *, uuid: str, fetcher: MetricsFetcher, storage: StorageService):
        self.uuid = uuid
        self.fetcher = fetcher
        self.storage = storage

    async def collect(self):
        metric = await self.fetcher.get_cpu_memory_metrics_1m(self.uuid)
        await self.storage.save_actual(metrics=metric, node=self.uuid)