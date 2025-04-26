from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timedelta
import aiohttp


class MetricsFetcher(ABC):
    @abstractmethod
    async def get_cpu_metrics_node_1m(self, uuid: str) -> pd.DataFrame:
        pass

    @abstractmethod
    async def get_memory_metrics_node_1m(self, uuid: str) -> dict:
        pass

    @abstractmethod
    async def get_cpu_memory_metrics_node_1m(self, uuid: str) -> dict:
        pass

    @abstractmethod
    async def get_cpu_memory_metrics_1m(self) -> dict:
        pass
# class MetricsFetcher:
#     def __init__(self, base_url: str):
#         self.base_url = base_url

#     async def _query(self, query: str, start_time: datetime, end_time: datetime, step: str = "1m") -> pd.DataFrame:
#         params = {
#             "query": query,
#             "start": start_time.timestamp(),
#             "end": end_time.timestamp(),
#             "step": step
#         }

#         async with aiohttp.ClientSession() as session:
#             async with session.get(f"{self.base_url}/api/v1/query_range", params=params) as response:
#                 response.raise_for_status()
#                 data = await response.json()

#                 if data["status"] != "success":
#                     raise ValueError(f"Query failed: {data.get('error', 'Unknown error')}")

#                 records = []
#                 for result in data["data"]["result"]:
#                     for value in result["values"]:
#                         timestamp = datetime.fromtimestamp(value[0])
#                         metric_value = float(value[1])
#                         records.append({
#                             "timestamp": timestamp,
#                             "value": metric_value
#                         })

#                 return pd.DataFrame(records)

#     async def get_cpu_metrics_1m(self, uuid: str) -> pd.DataFrame:
#         query = f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{uuid}"}}[1m]) * 100 / libvirt_domain_info_virtual_cpus{{uuid="{uuid}"}}'
#         return await self._query(query, None, None)

#     @abstractmethod
#     def get_memory_metrics(self, uuid: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
#         pass
