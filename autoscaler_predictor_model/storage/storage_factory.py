from storage.postgres_storage import PostgresStorage
from storage.influxdb_storage import InfluxDBStorage
from storage.storage_service import StorageService
from config import EnvConfig


class StorageFactory:
    @staticmethod
    def create_storage(config: EnvConfig) -> StorageService:
        if config.storage_type == 'influxdb':
            return InfluxDBStorage(**config.influxdb_config)
        else:
            return PostgresStorage(config.db_config)
