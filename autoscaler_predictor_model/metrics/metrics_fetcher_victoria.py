import aiohttp
import pandas as pd
from datetime import datetime, timedelta

from metrics.metrics_collector import MetricsFetcher

class VictoriaMetricsFetcher(MetricsFetcher):
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
                
                start_value = data["data"]["result"]["values"][0][1]
                end_value = data["data"]["result"]["values"][1][0]
                avg_value = (start_value + end_value) / 2
                timestamp = datetime.fromtimestamp(data["data"]["result"]["values"][1][0])  # Время конца минуты
                return {
                    "timestamp": timestamp,
                    "cpu": avg_value
                }


    async def get_cpu_metrics_1m(self, uuid: str) -> pd.DataFrame:
        query = f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{uuid}"}}[1m]) * 100 / libvirt_domain_info_virtual_cpus{{uuid="{uuid}"}}'
        return await self._query(query, None, None)

    def get_memory_metrics_1m(self, uuid: str) -> pd.DataFrame:
        # Замените на соответствующий запрос для памяти
        memory_query = f'rate(libvirt_domain_memory_stats_used_percent{{uuid="{uuid}"}}[1m])'
        return self._query(memory_query, None, None)
