import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAModel:
    def __init__(self, data_retention='4H'):
        self.historical_data = pd.DataFrame()
        self.data_retention = pd.to_timedelta(data_retention)
    
    def update_data(self, new_data):
        self.historical_data = pd.concat([self.historical_data, new_data])
        self.historical_data = self.historical_data.last(self.data_retention)
    
    def forecast(self):
        if len(self.historical_data) < 10:
            return None
        
        cpu_pred = self._forecast_metric('cpu')
        mem_pred = self._forecast_metric('memory')
        return {
            'cpu': cpu_pred,
            'memory': mem_pred
        }
    
    def _forecast_metric(self, metric):
        model = SARIMAX(
            self.historical_data[metric],
            order=(1,1,1),
            seasonal_order=(1,1,1,4)
        )
        results = model.fit(disp=False)
        return results.forecast(steps=1)[0] 
