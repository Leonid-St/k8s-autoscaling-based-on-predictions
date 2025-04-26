from models.metric_data import MetricData
from storage.storage_service import StorageService


class MetricPreparer:
    def __init__(self,storage:StorageService):
        async def on_new_data(self, metric_data: MetricData):
            real_cpu = metric_data.metrics['cpu'] * metric_data.metrics['node_count']
            rela_memory = metric_data.metrics['memory'] * metric_data.metrics['node_count']



