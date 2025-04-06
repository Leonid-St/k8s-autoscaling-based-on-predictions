import pandas as pd
import os
from datetime import datetime, timedelta

class MetricsStorage:
    def __init__(self, storage_path='./metrics_data', retention_period=timedelta(days=365)):
        self.storage_path = storage_path
        self.retention_period = retention_period
        self.observers = []  # Список наблюдателей
        os.makedirs(self.storage_path, exist_ok=True)

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, file_path):
        for observer in self.observers:
            observer.update(file_path)

    def _get_file_path(self, timestamp):
        date_str = timestamp.strftime('%Y-%m-%d')
        return os.path.join(self.storage_path, f'metrics_{date_str}.csv')

    def save_metrics(self, metrics: pd.DataFrame):
        timestamp = datetime.now()
        file_path = os.path.join(self.storage_path, f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv")
        metrics.to_csv(file_path, index=False)
        self.notify_observers(file_path)  # Уведомляем наблюдателей

    def _clean_old_files(self):
        now = datetime.now()
        for filename in os.listdir(self.storage_path):
            file_path = os.path.join(self.storage_path, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_time > self.retention_period:
                os.remove(file_path)

    def get_metrics(self, start_date, end_date):
        metrics = []
        current_date = start_date
        while current_date <= end_date:
            file_path = self._get_file_path(current_date)
            if os.path.exists(file_path):
                daily_metrics = pd.read_csv(file_path)
                daily_metrics['timestamp'] = pd.to_datetime(daily_metrics['timestamp'])
                metrics.append(daily_metrics)
            current_date += timedelta(days=1)
        
        if metrics:
            return pd.concat(metrics)
        return pd.DataFrame()

    def calculate_accuracy_over_time(self, start_date, end_date, model_type):
        metrics = self.get_metrics(start_date, end_date)
        if metrics.empty:
            return pd.DataFrame()

        # Проверяем наличие колонки 'model'
        if 'model' not in metrics.columns:
            raise ValueError("Column 'model' not found in metrics data")

        # Фильтруем метрики по типу модели
        metrics = metrics[metrics['model'] == model_type]
        if metrics.empty:
            return pd.DataFrame()

        metrics['absolute_error'] = abs(metrics['cpu_actual'] - metrics['cpu_predicted'])
        metrics['relative_error'] = metrics['absolute_error'] / metrics['cpu_actual']
        
        daily_accuracy = metrics.groupby(pd.Grouper(key='timestamp', freq='D')).agg({
            'absolute_error': 'mean',
            'relative_error': 'mean'
        }).reset_index()
        
        daily_accuracy['accuracy'] = 1 - daily_accuracy['relative_error']
        return daily_accuracy

    def save_errors(self, timestamp, node, absolute_error, relative_error):
        new_data = pd.DataFrame({
            'timestamp': [timestamp],
            'node': [node],
            'absolute_error': [absolute_error],
            'relative_error': [relative_error]
        })
        
        file_path = os.path.join(self.storage_path, 'errors.csv')
        
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path)
            combined_data = pd.concat([existing_data, new_data])
        else:
            combined_data = new_data
        
        combined_data.to_csv(file_path, index=False) 

    def get_errors(self, start_date, end_date):
        file_path = os.path.join(self.storage_path, 'errors.csv')
        if not os.path.exists(file_path):
            return pd.DataFrame()
        
        errors = pd.read_csv(file_path)
        return errors[(errors['timestamp'] >= start_date) & (errors['timestamp'] <= end_date)] 

    def get_latest_metrics(self) -> pd.DataFrame:
        files = sorted(os.listdir(self.storage_path), reverse=True)
        if not files:
            return pd.DataFrame()
        
        latest_file = os.path.join(self.storage_path, files[0])
        return pd.read_csv(latest_file) 
