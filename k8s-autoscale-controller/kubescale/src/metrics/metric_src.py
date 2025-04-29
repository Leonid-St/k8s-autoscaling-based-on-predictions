from utils.kubernetes_util import PodWrapper
from utils.prometheus_util import MetricWrapper
from typing import Dict, List, Union
import datetime
from abc import ABC, abstractmethod

class MetricSource(ABC):
    @abstractmethod
    def get_data(self, pods: List[PodWrapper], start_time: datetime.datetime, end_time: datetime.datetime) \
            -> Union[Dict[PodWrapper, MetricWrapper], MetricWrapper]:
        pass
