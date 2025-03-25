from xgboost import XGBRegressor
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

class XGBoostModel:
    def __init__(self, cluster_metrics):
        self.cpu_model = XGBRegressor()
        self.memory_model = XGBRegressor()
        self.cluster_metrics = cluster_metrics
    
    def partial_fit(self, df):
        if df.empty:
            raise ValueError("Empty training data")
        
        # Добавляем метрики кластера в фичи
        cluster_state = self.cluster_metrics.get_cluster_state()
        df['active_nodes'] = cluster_state['total_nodes']
        df['avg_cluster_cpu'] = cluster_state['average_cpu']
        df['avg_cluster_mem'] = cluster_state['average_memory']
        
        features = self._create_features(df)
        
        if 'cpu' in df.columns:
            if self.cpu_model.get_params()['n_estimators'] == 100:  # Дефолтные параметры = необученная модель
                self.cpu_model.fit(features, df['cpu'])
            else:
                self.cpu_model.partial_fit(features, df['cpu'])

        
            if 'memory' in df.columns:
                if not self.memory_model.get_params():
                    self.memory_model.fit(features, df['memory'])
                else:
                    self.memory_model.partial_fit(features, df['memory'])
    
    def predict(self, timestamp):
        features = self._create_features(pd.DataFrame({'timestamp': [timestamp]}))
        return {
            'cpu': self.cpu_model.predict(features)[0],
            'memory': self.memory_model.predict(features)[0]
        }
    
    def _create_features(self, df):
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'] >= 5
        df['minutes_since_midnight'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        return df[['hour', 'day_of_week', 'month', 'is_weekend', 'minutes_since_midnight']] 
