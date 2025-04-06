from influxdb_client import InfluxDBClient
from datetime import datetime
import pandas as pd
from storage.storage_service import StorageService

class InfluxDBStorage(StorageService):
    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket

    def save_prediction(self, timestamp: datetime, node: str, model_type: str, prediction: dict):
        write_api = self.client.write_api()
        data = f"predictions,node={node},model_type={model_type} cpu={prediction.get('cpu', 0)},memory={prediction.get('memory', 0)} {int(timestamp.timestamp())}"
        write_api.write(self.bucket, record=data)

    def save_actual(self, timestamp: datetime, node: str, metrics: dict):
        write_api = self.client.write_api()
        data = f"actuals,node={node} cpu={metrics.get('cpu', 0)},memory={metrics.get('memory', 0)} {int(timestamp.timestamp())}"
        write_api.write(self.bucket, record=data)

    def save_error(self, timestamp: datetime, node: str, model_type: str, error_metrics: dict):
        write_api = self.client.write_api()
        data = f"errors,node={node},model_type={model_type} mse={error_metrics.get('mse', 0)},mae={error_metrics.get('mae', 0)} {int(timestamp.timestamp())}"
        write_api.write(self.bucket, record=data)

    def get_predictions(self, start_date: datetime, end_date: datetime, node: str = None, model_type: str = None) -> pd.DataFrame:
        query = f'from(bucket: "{self.bucket}") |> range(start: {start_date.isoformat()}, stop: {end_date.isoformat()})'
        query += ' |> filter(fn: (r) => r._measurement == "predictions")'
        if node:
            query += f' |> filter(fn: (r) => r.node == "{node}")'
        if model_type:
            query += f' |> filter(fn: (r) => r.model_type == "{model_type}")'
        return self._query_to_df(query)

    def get_actuals(self, start_date: datetime, end_date: datetime, node: str = None) -> pd.DataFrame:
        query = f'from(bucket: "{self.bucket}") |> range(start: {start_date.isoformat()}, stop: {end_date.isoformat()})'
        query += ' |> filter(fn: (r) => r._measurement == "actuals")'
        if node:
            query += f' |> filter(fn: (r) => r.node == "{node}")'
        return self._query_to_df(query)

    def get_errors(self, start_date: datetime, end_date: datetime, node: str = None, model_type: str = None) -> pd.DataFrame:
        query = f'from(bucket: "{self.bucket}") |> range(start: {start_date.isoformat()}, stop: {end_date.isoformat()})'
        query += ' |> filter(fn: (r) => r._measurement == "errors")'
        if node:
            query += f' |> filter(fn: (r) => r.node == "{node}")'
        if model_type:
            query += f' |> filter(fn: (r) => r.model_type == "{model_type}")'
        return self._query_to_df(query)

    def _query_to_df(self, query: str) -> pd.DataFrame:
        result = self.client.query_api().query(query)
        records = []
        for table in result:
            for record in table.records:
                records.append((record.get_time(), record.values))
        return pd.DataFrame(records, columns=['timestamp', 'values']) 
