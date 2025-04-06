import aiohttp
import pandas as pd
from datetime import datetime, timedelta

class VictoriaMetricsCollector(MetricsCollector):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def _query(self, query: str, start_time: datetime, end_time: datetime, step: str = "1m") -> pd.DataFrame:
        params = {
            "query": query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": step
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/v1/query_range", params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data["status"] != "success":
                    raise ValueError(f"Query failed: {data.get('error', 'Unknown error')}")
                
                records = []
                for result in data["data"]["result"]:
                    # Берем два последних значения из массива values
                    if len(result["values"]) >= 2:
                        start_value = float(result["values"][-2][1])  # Значение на начало минуты
                        end_value = float(result["values"][-1][1])    # Значение на конец минуты
                        avg_value = (start_value + end_value) / 2     # Среднее значение
                        
                        timestamp = datetime.fromtimestamp(result["values"][-1][0])  # Время конца минуты
                        records.append({
                            "timestamp": timestamp,
                            "value": avg_value
                        })
                
                return pd.DataFrame(records)

    async def get_cpu_metrics(self, uuid: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        query = f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{uuid}"}}[1m]) * 100 / libvirt_domain_info_virtual_cpus{{uuid="{uuid}"}}'
        return await self._query(query, start_time, end_time)

    def get_memory_metrics(self, uuid: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        # Замените на соответствующий запрос для памяти
        query = f'your_memory_query{{uuid="{uuid}"}}'
        return self._query(query, start_time, end_time)
