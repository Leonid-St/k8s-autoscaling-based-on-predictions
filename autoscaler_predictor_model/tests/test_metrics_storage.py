import pytest
import pandas as pd
from datetime import datetime, timedelta
from utils.metrics_storage import MetricsStorage

@pytest.fixture
def metrics_storage():
    return MetricsStorage()

def test_save_metrics(metrics_storage):
    timestamp = datetime.now()
    metrics_storage.save_metrics(timestamp, 'node1', 0.5, 0.4)
    metrics = metrics_storage.get_metrics(timestamp, timestamp)
    assert not metrics.empty
    assert metrics.iloc[0]['cpu_actual'] == 0.5

def test_calculate_accuracy_over_time(metrics_storage):
    timestamp = datetime.now()
    metrics_storage.save_metrics(timestamp, 'node1', 0.5, 0.4)
    accuracy = metrics_storage.calculate_accuracy_over_time(timestamp, timestamp)
    assert not accuracy.empty
    assert 'accuracy' in accuracy.columns
