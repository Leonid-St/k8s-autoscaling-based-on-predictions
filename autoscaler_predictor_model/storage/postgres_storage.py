from abc import ABC

import psycopg2
import pandas as pd
from datetime import datetime
import json
from sqlalchemy import text

from models.errors_metrics import ErrorsMetrics
from models.metric_data import MetricData
from storage.storage_service import StorageService
import logging
import uvicorn

logger = logging.getLogger("uvicorn.error")


def get_valid_table_name(uuid: str, table_type: str) -> str:
    return f"{table_type}_node_{uuid.replace('-', '')}"


class PostgresStorage(StorageService):
    async def get_errors(self, *, start_date: datetime, end_date: datetime, node: str = None, model_type: str = None,
                         error_metrics: dict) -> MetricData | None:
        pass

    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = self._get_connection()
        self.actual_table_created = False
        self.predicted_table_created = False
        self.error_table_created = False

        # self._init_tables()

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

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def _ensure_connection(self):
        if self.conn.closed:
            self.conn = self._get_connection()

    async def _create_table_if_not_exists(self, table_name: str, table_type: str):
        with self.conn.cursor() as cur:
            if table_type == "actual":
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        timestamp TIMESTAMPTZ PRIMARY KEY,
                        actual_cpu FLOAT,
                        actual_memory FLOAT,
                        actual_pod_count INTEGER
                    )
                """)
                self.actual_table_created = True
            elif table_type == "predicted":
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        timestamp TIMESTAMPTZ PRIMARY KEY,
                        predicted_cpu FLOAT,
                        predicted_memory FLOAT,
                    )
                """)
                self.predicted_table_created = True
            elif table_type == "error":
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp TIMESTAMPTZ PRIMARY KEY,
                    end_time TIMESTAMPTZ,
                    cpu_mse FLOAT,
                    cpu_mae FLOAT,
                    cpu_rmse FLOAT,
                    memory_mse FLOAT,
                    memory_mae FLOAT,
                    memory_rmse FLOAT
                )
                """)
                self.error_table_created = False
            cur.execute(f"""
                SELECT create_hypertable('{table_name}', 'timestamp',if_not_exists => TRUE);
            """)
            self.conn.commit()
            logger.info("table was created with table_name:" + table_name)

    async def save_prediction(self, *,
                              timestamp: datetime,
                              node: str,
                              prediction: dict,
                              ):
        table_name = get_valid_table_name(node, "predicted")
        if not self.predicted_table_created:
            await self._create_table_if_not_exists(table_name, "predicted")

        with self.conn.cursor() as cur:
            try:
                cur.execute(f"""
                    INSERT INTO {table_name} (timestamp, predicted_cpu, predicted_memory)
                    VALUES (%s, %s, %s)
                """, (timestamp, float(prediction.get('cpu')), float(prediction.get('memory'))))
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                logger.error(f"Error saving predicted metrics: {e}")
                raise
        logger.info("metric predicted saved in postgres for: " + node)

    async def save_actual(self, *, node: str, metrics: MetricData):
        self._ensure_connection()
        table_name = get_valid_table_name(node, "actual")
        if not self.actual_table_created:
            await self._create_table_if_not_exists(table_name, "actual")

        with self.conn.cursor() as cur:
            try:
                cur.execute(f"""
                    INSERT INTO {table_name} (timestamp, actual_cpu, actual_memory,actual_pod_count)
                    VALUES (%s, %s, %s)
                """, (metrics.timestamp,
                      float(metrics.metrics.get('cpu')),
                      float(metrics.metrics.get('memory')),
                      int(metrics.metrics.get('pod_count'))
                      ))
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                logger.error(f"Error saving actual metrics: {e}")
                raise
        logger.info("metric actual saved in postgres for: " + node)

    async def save_error(self, *,
                         node: str,
                         error_metrics: ErrorsMetrics,
                         ):
        table_name = get_valid_table_name(node, "error")
        if not self.error_table_created:
            await self._create_table_if_not_exists(table_name, "error")
        with self.conn.cursor() as cur:
            try:
                cur.execute(f"""
                    INSERT INTO {table_name} (
                        timestamp,
                        end_time,
                        cpu_mse ,
                        cpu_mae, 
                        cpu_rmse, 
                        memory_mse, 
                        memory_mae, 
                        memory_rmse )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    error_metrics.start_time,
                    error_metrics.end_time,
                    error_metrics.cpu.mse,
                    error_metrics.cpu.mae,
                    error_metrics.cpu.rmse,
                    error_metrics.memory.mse,
                    error_metrics.memory.mae,
                    error_metrics.memory.rmse
                ))
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                logger.error(f"Error saving error metrics: {e}")
        logger.info("metric error saved in postgres for: " + node)

    async def get_prediction(self, *,
                             timestamp: datetime,
                             node: str = None,
                             model_type: str = None) -> MetricData | None:
        if not node:
            raise ValueError("Node must be specified")

        table_name = get_valid_table_name(node, "predicted")
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT timestamp, predicted_cpu, predicted_memory
                FROM {table_name}
                WHERE timestamp = %s
            """, (timestamp,))
            result = cur.fetchone()
            if result:
                return MetricData(timestamp=result[0], metrics={'cpu': result[1], 'memory': result[2]})
            return None

    async def get_latest_prediction(self, *, node: str = None) -> MetricData | None:
        """
        Получает последнее предсказание для указанной ноды из базы данных.
        """
        try:
            table_name = get_valid_table_name(node, "predicted")
            with self.conn.cursor() as cur:
                cur.execute(f"""
                        SELECT timestamp, predicted_cpu,predicted_memory
                        FROM {table_name}
                        ORDER BY timestamp DESC
                        LIMIT 1"""
                )
                result = result.fetchone()
                if result:
                    return MetricData(timestamp=result[0], metrics={'cpu': result[1], 'memory': result[2]})
                return None
        except Exception as e:
            logger.error(f"Error getting last predicted error metrics: {e}")

    async def get_actual(self, *,
                         node: str,
                         timestamp: datetime,
                         ) -> MetricData | None:
        table_name = get_valid_table_name(node, "actual")
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT timestamp, actual_cpu, actual_memory
                FROM {table_name}
                WHERE timestamp = %s
            """, (timestamp,))
            result = cur.fetchone()
            if result:
                return MetricData(timestamp=result[0], metrics={'cpu': result[1], 'memory': result[2]})
            return None

    # def get_errors(self,*, start_date: datetime,
    #                end_date: datetime,
    #                node: str = None,
    #                model_type: str = None,
    #                ) -> pd.DataFrame:
    #     if not node:
    #         raise ValueError("Node must be specified")
    #
    #     table_name = get_valid_table_name(node, "error")
    #     query = f"""
    #         SELECT timestamp, mse, mae
    #         FROM {table_name}
    #         WHERE timestamp BETWEEN %s AND %s
    #     """
    #     params = [start_date, end_date]
    #
    #     return pd.read_sql(query, self.conn, params=params)

    # async def get_actual_range(self,*, start_time: datetime,
    #                            end_time: datetime,
    #                            node: str,
    #                            ) -> pd.DataFrame:
    #     table_name = get_valid_table_name(node, "actual")
    #     with self.conn.cursor() as cur:
    #         cur.execute(f"""
    #             SELECT timestamp, actual_cpu, actual_memory
    #             FROM {table_name}
    #             WHERE timestamp BETWEEN %s AND %s
    #         """, (start_time, end_time))
    #         results = cur.fetchall()
    #         if results:
    #             return pd.DataFrame(results, columns=['timestamp', 'cpu', 'memory'])
    #         return pd.DataFrame()

    # async def get_predictions_range(self,*, start_time: datetime, end_time: datetime, node: str) -> pd.DataFrame:
    #     table_name = get_valid_table_name(node, "predicted")
    #     with self.conn.cursor() as cur:
    #         cur.execute(f"""
    #             SELECT timestamp, predicted_cpu, predicted_memory
    #             FROM {table_name}
    #             WHERE timestamp BETWEEN %s AND %s
    #         """, (start_time, end_time))
    #         results = cur.fetchall()
    #         if results:
    #             return pd.DataFrame(results, columns=['timestamp', 'cpu', 'memory'])
    #         return pd.DataFrame()

    # async def save_error(self, *,
    #                      start_time: datetime,
    #                      end_time: datetime,
    #                      node: str,
    #                      model_type: str,
    #                      error_metrics: dict):
    #     table_name = get_valid_table_name(node, "error")
    #     await self._create_table_if_not_exists(table_name, "error")
    #
    #     with self.conn.cursor() as cur:
    #         cur.execute(f"""
    #             INSERT INTO {table_name} (timestamp, end_time, mse, mae)
    #             VALUES (%s, %s, %s, %s)
    #         """, (start_time, end_time, error_metrics.get('rmse'), error_metrics.get('mae')))
    #         self.conn.commit()
