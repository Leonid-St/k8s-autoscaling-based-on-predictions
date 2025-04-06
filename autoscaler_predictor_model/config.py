import os
from typing import Optional

class EnvConfig:
    def __init__(self):
        self.uuid_node = self._get_env_var("UUID_NODE", required=True)
        self.db_config = self._get_db_config()

    def _get_env_var(self, var_name: str, required: bool = False) -> Optional[str]:
        value = os.getenv(var_name)
        if required and not value:
            raise ValueError(f"Environment variable {var_name} is required but not set.")
        return value

    def _get_db_config(self) -> dict:
        return {
            'dbname': self._get_env_var("DB_NAME", required=True),
            'user': self._get_env_var("DB_USER", required=True),
            'password': self._get_env_var("DB_PASSWORD", required=True),
            'host': self._get_env_var("DB_HOST", required=True),
            'port': self._get_env_var("DB_PORT", required=True)
        } 
