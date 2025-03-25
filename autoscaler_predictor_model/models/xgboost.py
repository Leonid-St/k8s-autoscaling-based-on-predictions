from xgboost import XGBRegressor
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

class XGBoostModel:
    def __init__(self, cluster_metrics):
        self.cpu_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            n_jobs=-1  # Используем все ядра CPU
        )
        self.memory_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            n_jobs=-1  # Используем все ядра CPU
        )
        self.cluster_metrics = cluster_metrics
        self.cpu_scaler = StandardScaler()
        self.memory_scaler = StandardScaler()
        self._cpu_fitted = False
        self._memory_fitted = False
    
    def partial_fit(self, df):
        if df.empty:
            raise ValueError("Empty training data")
        
        # Используем .loc для избежания предупреждений
        cluster_state = self.cluster_metrics.get_cluster_state()
        df = df.copy()
        df.loc[:, 'active_nodes'] = cluster_state['total_nodes']
        df.loc[:, 'avg_cluster_cpu'] = cluster_state['average_cpu']
        df.loc[:, 'avg_cluster_mem'] = cluster_state['average_memory']
        
        features = self._create_features(df)
        
        if 'cpu' in df.columns:
            cpu_target = self.cpu_scaler.fit_transform(df[['cpu']])
            if self._cpu_fitted:
                self.cpu_model.fit(features, cpu_target, xgb_model=self.cpu_model.get_booster())
            else:
                self.cpu_model.fit(features, cpu_target)
                self._cpu_fitted = True
        
        if 'memory' in df.columns:
            memory_target = self.memory_scaler.fit_transform(df[['memory']])
            if self._memory_fitted:
                self.memory_model.fit(features, memory_target, xgb_model=self.memory_model.get_booster())
            else:
                self.memory_model.fit(features, memory_target)
                self._memory_fitted = True
    
    def predict(self, timestamp):
        features = self._create_features(pd.DataFrame({'timestamp': [timestamp]}))
        cpu_pred = self.cpu_model.predict(features)
        memory_pred = self.memory_model.predict(features)
        
        # Обратная нормализация данных
        return {
            'cpu': self.cpu_scaler.inverse_transform([cpu_pred])[0][0],
            'memory': self.memory_scaler.inverse_transform([memory_pred])[0][0]
        }
    
    def _create_features(self, df):
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'] >= 5
        df['minutes_since_midnight'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        return df[['hour', 'day_of_week', 'month', 'is_weekend', 'minutes_since_midnight']]
