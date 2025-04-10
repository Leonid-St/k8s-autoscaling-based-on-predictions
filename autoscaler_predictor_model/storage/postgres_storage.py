import psycopg2
import pandas as pd
from datetime import datetime
from storage.storage_service import StorageService
import logging
import uvicorn
logger = logging.getLogger("uvicorn")

def get_valid_table_name(uuid: str, table_type: str) -> str:
    return f"{table_type}_node_{uuid.replace('-', '')}"

class PostgresStorage(StorageService):
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        #self._init_tables()

    # def _init_tables(self):
    #     with self.conn.cursor() as cur:
    #         # Создаем таблицы, если они не существуют
    #         cur.execute("""
    #             CREATE EXTENSION IF NOT EXISTS timescaledb;
    #         """)
    #         cur.execute("""
    #             CREATE TABLE IF NOT EXISTS predictions (
    #                 id SERIAL PRIMARY KEY,
    #                 timestamp TIMESTAMPTZ NOT NULL,
    #                 node VARCHAR(255) NOT NULL,
    #                 model_type VARCHAR(255) NOT NULL,
    #                 cpu FLOAT,
    #                 memory FLOAT
    #             )
    #         """)
    #         cur.execute("""
    #             CREATE TABLE IF NOT EXISTS actuals (
    #                 id SERIAL PRIMARY KEY,
    #                 timestamp TIMESTAMPTZ NOT NULL,
    #                 node VARCHAR(255) NOT NULL,
    #                 cpu FLOAT,
    #                 memory FLOAT
    #             )
    #         """)
    #         cur.execute("""
    #             CREATE TABLE IF NOT EXISTS errors (
    #                 id SERIAL PRIMARY KEY,
    #                 timestamp TIMESTAMPTZ NOT NULL,
    #                 node VARCHAR(255) NOT NULL,
    #                 model_type VARCHAR(255) NOT NULL,
    #                 mse FLOAT,
    #                 mae FLOAT
    #             )
    #         """)
    #         self.conn.commit()

    #     # Преобразуем таблицы в гипертаблицы
    #     with self.conn.cursor() as cur:
    #         cur.execute("""
    #             SELECT create_hypertable('predictions', by_range('timestamp'));
    #         """)
    #         cur.execute("""
    #             SELECT create_hypertable('actuals', by_range('timestamp'));
    #         """)
    #         cur.execute("""
    #             SELECT create_hypertable('errors', by_range('timestamp'));
    #         """)
    #         self.conn.commit()

    def _create_table_if_not_exists(self, table_name: str, table_type: str):
        with self.conn.cursor() as cur:
            if table_type == "actual":
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        actual_cpu FLOAT,
                        actual_memory FLOAT
                    )
                """)
            elif table_type == "predicted":
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        predicted_cpu FLOAT,
                        predicted_memory FLOAT
                    )
                """)
            elif table_type == "error":
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        mse FLOAT,
                        mae FLOAT
                    )
                """)
            cur.execute(f"""
                SELECT create_hypertable('{table_name}', by_range('timestamp'));
            """)
            self.conn.commit()
            print("table was created with table_name:"+table_name)

    async def save_prediction(self, timestamp: datetime, node: str, model_type: str, prediction: dict):
        table_name = get_valid_table_name(node, "predicted")
        await self._create_table_if_not_exists(table_name, "predicted")
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {table_name} (timestamp, predicted_cpu, predicted_memory)
                VALUES (%s, %s, %s)
            """, (timestamp, prediction.get('cpu'), prediction.get('memory')))
            await self.conn.commit()

    async def save_actual(self,*, node: str, metrics: dict):
        table_name = get_valid_table_name(node, "actual")
        await self._create_table_if_not_exists(table_name, "actual")
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {table_name} (timestamp, actual_cpu, actual_memory)
                VALUES (%s, %s, %s)
            """, (metrics.get('timestamp'), metrics.get('cpu'), metrics.get('memory')))
            await self.conn.commit()
        logger.info("metric saved in postgres for: "+node)

    def save_error(self, timestamp: datetime, node: str, model_type: str, error_metrics: dict):
        table_name = get_valid_table_name(node, "error")
        self._create_table_if_not_exists(table_name, "error")
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {table_name} (timestamp, mse, mae)
                VALUES (%s, %s, %s)
            """, (timestamp, error_metrics.get('mse'), error_metrics.get('mae')))
            self.conn.commit()

    def get_predictions(self, start_date: datetime, end_date: datetime, node: str = None, model_type: str = None) -> pd.DataFrame:
        if not node:
            raise ValueError("Node must be specified")
        
        table_name = get_valid_table_name(node, "predicted")
        query = f"""
            SELECT timestamp, predicted_cpu, predicted_memory
            FROM {table_name}
            WHERE timestamp BETWEEN %s AND %s
        """
        params = [start_date, end_date]
        
        return pd.read_sql(query, self.conn, params=params)

    def get_actuals(self, start_date: datetime, end_date: datetime, node: str = None) -> pd.DataFrame:
        if not node:
            raise ValueError("Node must be specified")
        
        table_name = get_valid_table_name(node, "actual")
        query = f"""
            SELECT timestamp, actual_cpu, actual_memory
            FROM {table_name}
            WHERE timestamp BETWEEN %s AND %s
        """
        params = [start_date, end_date]
        
        return pd.read_sql(query, self.conn, params=params)

    def get_errors(self, start_date: datetime, end_date: datetime, node: str = None, model_type: str = None) -> pd.DataFrame:
        if not node:
            raise ValueError("Node must be specified")
        
        table_name = get_valid_table_name(node, "error")
        query = f"""
            SELECT timestamp, mse, mae
            FROM {table_name}
            WHERE timestamp BETWEEN %s AND %s
        """
        params = [start_date, end_date]
        
        return pd.read_sql(query, self.conn, params=params) 
