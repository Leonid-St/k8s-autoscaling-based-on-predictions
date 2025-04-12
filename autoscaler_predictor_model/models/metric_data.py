from datetime import datetime
from typing import Dict


class MetricData:
    def __init__(self, timestamp: datetime, metrics: Dict[str, float]):
        self.timestamp = timestamp
        self.metrics = metrics

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            **self.metrics
        }
