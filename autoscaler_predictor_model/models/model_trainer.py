import pandas as pd

from storage.storage_service import StorageService
class ModelTrainer:
    def __init__(self, metrics_storage: StorageService , model):
        self.metrics_storage = metrics_storage
        self.model = model
        self.metrics_storage.add_observer(self)  # Регистрируем себя как наблюдателя

    def get_latest_metric():
        
        
    def update(self):
        # Обрабатываем новый файл
        #latest_metrics = pd.read_csv(file_path)
        self.metrics_storage.
        if not latest_metrics.empty:
            self.model.fit(latest_metrics)
            print("Model updated with new data") 
