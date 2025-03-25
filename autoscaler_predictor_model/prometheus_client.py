import requests
import pandas as pd
from datetime import datetime, timedelta


class PrometheusDataCollector:
    def __init__(self, prometheus_url, query_cpu, query_memory):
        self.prometheus_url = prometheus_url
        self.query_cpu = query_cpu
        self.query_memory = query_memory
        self.last_update = datetime.now() - timedelta(minutes=15)

    def _query_prometheus(self, query):
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query, 'time': self.last_update.timestamp()}
            )
            response.raise_for_status()
            return response.json()['data']['result']
        except Exception as e:
            print(f"Error querying Prometheus: {str(e)}")
            return []

    def get_new_data(self):
        cpu_data = self._process_metrics(self._query_prometheus(self.query_cpu), 'cpu')
        mem_data = self._process_metrics(self._query_prometheus(self.query_memory), 'memory')

        self.last_update = datetime.now()
        return pd.merge(cpu_data, mem_data, on='timestamp', how='outer').fillna(method='ffill')

    def _process_metrics(self, data, metric_name):
        records = []
        for result in data:
            timestamp = datetime.fromtimestamp(result['value'][0])
            value = float(result['value'][1])
            records.append({
                'timestamp': timestamp,
                metric_name: value
            })
        return pd.DataFrame(records)
