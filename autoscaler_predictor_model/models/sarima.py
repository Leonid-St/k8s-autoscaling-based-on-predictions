import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAModel:
    def __init__(self, data_retention='4h'):
        self.historical_data = pd.DataFrame()
        self.data_retention = pd.to_timedelta(data_retention.replace('H', 'h'))
    
    def update_data(self, new_data):
        combined = pd.concat([self.historical_data, new_data])
        combined = combined.sort_index().last(self.data_retention)
        self.historical_data = combined.drop_duplicates()
    
    def forecast(self):
        if self.historical_data.empty:
            raise ValueError("No historical data available")
            
        if len(self.historical_data) < 10:
            raise ValueError("Minimum 10 data points required for SARIMA")
        
        try:
            cpu_pred = self._forecast_metric('cpu')
            mem_pred = self._forecast_metric('memory')
        except Exception as e:
            raise RuntimeError(f"SARIMA forecasting failed: {str(e)}")
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
