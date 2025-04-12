from datetime import datetime
from dataclasses import dataclass

@dataclass
class ErrorsProba:
    mse: float
    mae: float
    rmse: float

@dataclass
class ErrorsMetrics:
    start_time: datetime
    end_time: datetime
    cpu: ErrorsProba
    memory: ErrorsProba 
