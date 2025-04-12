from typing import Dict

import aiohttp
import pandas as pd
from datetime import datetime, timedelta

from metrics.metrics_fetcher import MetricsFetcher
import logging
import uvicorn

logger = logging.getLogger('uvicorn.error')


class VictoriaMetricsFetcher(MetricsFetcher):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def _query(self, *,
                     query: str,
                     start_time: datetime | None = None,
                     end_time: datetime | None = None,
                     step: str = "1m",
                     ) -> dict[str, float | datetime]:
        params = {
            "query": query,
            "step": step
        }

        if start_time is not None:
            params["start"] = start_time.timestamp()

        if end_time is not None:
            params["end"] = end_time.timestamp()

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/v1/query_range", params=params) as response:
                response.raise_for_status()
                data = await response.json()

                if data["status"] != "success":
                    raise ValueError(f"Query failed: {data.get('error', 'Unknown error')}")

                start_value = data["data"]["result"][0]["values"][0][1]
                end_value = data["data"]["result"][0]["values"][1][1]
                avg_value = (float(start_value) + float(end_value)) / 2
                timestamp = datetime.fromtimestamp(int(data["data"]["result"][0]["values"][1][0]))  # Время конца минуты
                return {
                    "timestamp": timestamp,
                    "value": avg_value
                }

    async def get_cpu_metrics_1m(self, uuid: str) -> dict[str, float | datetime]:
        query = f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{uuid}"}}[1m]) * 100 / libvirt_domain_info_virtual_cpus{{uuid="{uuid}"}}'
        return await self._query(query=query)

    async def get_memory_metrics_1m(self, uuid: str) -> dict[str, float | datetime]:
        # Замените на соответствующий запрос для памяти
        memory_query = f'rate(libvirt_domain_memory_stats_used_percent{{uuid="{uuid}"}}[1m])'
        return await self._query(query=memory_query)

    async def get_cpu_memory_metrics_1m(self, uuid: str):
        cpu = await self.get_cpu_metrics_1m(uuid)
        memory = await self.get_memory_metrics_1m(uuid)
        logger.info("metric fetched for: " + uuid)
        return {
            "timestamp": cpu["timestamp"],
            "cpu": cpu["value"],
            "memory": memory["value"],
        }
