import pandas as pd
from datetime import datetime, timedelta

class MetricsComparator:
    def __init__(self, data_retention='4H'):
        self.actual_metrics = pd.DataFrame(columns=['timestamp', 'node', 'cpu'])
        self.predicted_metrics = pd.DataFrame(columns=['timestamp', 'node', 'cpu'])
        self.data_retention = pd.to_timedelta(data_retention.replace('H', 'h'))

    def _clean_old_data(self):
        """Удаляем данные старше data_retention"""
        now = datetime.now()
        self.actual_metrics = self.actual_metrics[self.actual_metrics['timestamp'] > now - self.data_retention]
        self.predicted_metrics = self.predicted_metrics[self.predicted_metrics['timestamp'] > now - self.data_retention]

    def add_actual_metric(self, timestamp: datetime, node: str, cpu: float):
        new_row = pd.DataFrame({'timestamp': [timestamp], 'node': [node], 'cpu': [cpu]})
        self.actual_metrics = pd.concat([self.actual_metrics, new_row], ignore_index=True)
        self._clean_old_data()

    def add_predicted_metric(self, timestamp: datetime, node: str, cpu: float):
        new_row = pd.DataFrame({'timestamp': [timestamp], 'node': [node], 'cpu': [cpu]})
        self.predicted_metrics = pd.concat([self.predicted_metrics, new_row], ignore_index=True)
        self._clean_old_data()

    def get_comparison(self, node: str = None):
        if node:
            actual = self.actual_metrics[self.actual_metrics['node'] == node]
            predicted = self.predicted_metrics[self.predicted_metrics['node'] == node]
        else:
            actual = self.actual_metrics
            predicted = self.predicted_metrics

        merged = pd.merge(actual, predicted, on=['timestamp', 'node'], suffixes=('_actual', '_predicted'))
        return merged

    def calculate_errors(self, node: str = None):
        comparison = self.get_comparison(node)
        if comparison.empty:
            return {}

        comparison['absolute_error'] = abs(comparison['cpu_actual'] - comparison['cpu_predicted'])
        comparison['relative_error'] = comparison['absolute_error'] / comparison['cpu_actual']

        return {
            'mean_absolute_error': comparison['absolute_error'].mean(),
            'mean_relative_error': comparison['relative_error'].mean(),
            'max_absolute_error': comparison['absolute_error'].max(),
            'max_relative_error': comparison['relative_error'].max()
        } 
