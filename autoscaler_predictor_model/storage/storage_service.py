from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

from models.metric_data import MetricData
from models.errors_metrics import ErrorsMetrics


class StorageService(ABC):
    @abstractmethod
    def save_prediction(self, *, timestamp: datetime, node: str,
                        prediction: dict):
        pass

    @abstractmethod
    async def save_actual(self, *, node: str, metrics: MetricData):
        pass

    @abstractmethod
    def save_error(self, *,
                   node: str,
                   error_metrics: ErrorsMetrics):
        pass

    @abstractmethod
    async def get_prediction(self, *,
                             timestamp: datetime,
                             node: str = None,
                             ) -> MetricData | None:
        pass

    @abstractmethod
    async def get_actual(self, *, node: str, timestamp: datetime) -> MetricData | None:
        pass

    @abstractmethod
    async def get_errors(self, *,
                         start_date: datetime,
                         end_date: datetime,
                         node: str = None,
                         error_metrics: dict,
                         ) -> MetricData | None:
        pass

    @abstractmethod
    async def get_latest_prediction(self, *, node: str) -> MetricData | None:
        pass
