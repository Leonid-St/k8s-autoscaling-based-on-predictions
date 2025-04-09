import psycopg2
import pandas as pd
from datetime import datetime
from storage.storage_service import StorageService

class PostgresStorage(StorageService):
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self._init_tables()

    def _init_tables(self):
        with self.conn.cursor() as cur:
            # Создаем таблицы, если они не существуют
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS timescaledb;
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    node VARCHAR(255) NOT NULL,
                    model_type VARCHAR(255) NOT NULL,
                    cpu FLOAT,
                    memory FLOAT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS actuals (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    node VARCHAR(255) NOT NULL,
                    cpu FLOAT,
                    memory FLOAT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    node VARCHAR(255) NOT NULL,
                    model_type VARCHAR(255) NOT NULL,
                    mse FLOAT,
                    mae FLOAT
                )
            """)
            self.conn.commit()

        # Преобразуем таблицы в гипертаблицы
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT create_hypertable('predictions', by_range('timestamp'));
            """)
            cur.execute("""
                SELECT create_hypertable('actuals', by_range('timestamp'));
            """)
            cur.execute("""
                SELECT create_hypertable('errors', by_range('timestamp'));
            """)
            self.conn.commit()

    def save_prediction(self, timestamp: datetime, node: str, model_type: str, prediction: dict):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions (timestamp, node, model_type, cpu, memory)
                VALUES (%s, %s, %s, %s, %s)
            """, (timestamp, node, model_type, prediction.get('cpu'), prediction.get('memory')))
            self.conn.commit()

    def save_actual(self, timestamp: datetime, node: str, metrics: dict):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO actuals (timestamp, node, cpu, memory)
                VALUES (%s, %s, %s, %s)
            """, (timestamp, node, metrics.get('cpu'), metrics.get('memory')))
            self.conn.commit()

    def save_error(self, timestamp: datetime, node: str, model_type: str, error_metrics: dict):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO errors (timestamp, node, model_type, mse, mae)
                VALUES (%s, %s, %s, %s, %s)
            """, (timestamp, node, model_type, error_metrics.get('mse'), error_metrics.get('mae')))
            self.conn.commit()

    def get_predictions(self, start_date: datetime, end_date: datetime, node: str = None, model_type: str = None) -> pd.DataFrame:
        query = """
            SELECT timestamp, node, model_type, cpu, memory
            FROM predictions
            WHERE timestamp BETWEEN %s AND %s
        """
        params = [start_date, end_date]
        
        if node:
            query += " AND node = %s"
            params.append(node)
        if model_type:
            query += " AND model_type = %s"
            params.append(model_type)
        
        return pd.read_sql(query, self.conn, params=params)

    def get_actuals(self, start_date: datetime, end_date: datetime, node: str = None) -> pd.DataFrame:
        query = """
            SELECT timestamp, node, cpu, memory
            FROM actuals
            WHERE timestamp BETWEEN %s AND %s
        """
        params = [start_date, end_date]
        
        if node:
            query += " AND node = %s"
            params.append(node)
        
        return pd.read_sql(query, self.conn, params=params)

    def get_errors(self, start_date: datetime, end_date: datetime, node: str = None, model_type: str = None) -> pd.DataFrame:
        query = """
            SELECT timestamp, node, model_type, mse, mae
            FROM errors
            WHERE timestamp BETWEEN %s AND %s
        """
        params = [start_date, end_date]
        
        if node:
            query += " AND node = %s"
            params.append(node)
        if model_type:
            query += " AND model_type = %s"
            params.append(model_type)
        
        return pd.read_sql(query, self.conn, params=params) 
