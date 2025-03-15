from xgboost import XGBRegressor
import pandas as pd

class XGBoostModel:
    def __init__(self):
        self.cpu_model = XGBRegressor()
        self.memory_model = XGBRegressor()
    
    def partial_fit(self, df):
        features = self._create_features(df)
        
        if 'cpu' in df.columns:
            if not self.cpu_model.get_params():
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
