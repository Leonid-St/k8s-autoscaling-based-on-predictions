from storage.postgres_storage import PostgresStorage
from storage.influxdb_storage import InfluxDBStorage
from storage.storage_service import StorageService

class StorageFactory:
    @staticmethod
    def create_storage(config: dict) -> StorageService:
        storage_type = config.get('STORAGE_TYPE', 'postgres')
        
        if storage_type == 'influxdb':
            return InfluxDBStorage(
                url=config['INFLUXDB_URL'],
                token=config['INFLUXDB_TOKEN'],
                org=config['INFLUXDB_ORG'],
                bucket=config['INFLUXDB_BUCKET']
            )
        else:
            return PostgresStorage({
                'dbname': config['POSTGRES_DB_NAME'],
                'user': config['POSTGRES_USER'],
                'password': config['POSTGRES_PASSWORD'],
                'host': config['POSTGRES_HOST'],
                'port': config['POSTGRES_PORT']
            }) 
