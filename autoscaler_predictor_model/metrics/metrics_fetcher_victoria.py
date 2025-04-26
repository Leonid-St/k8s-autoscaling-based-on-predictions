from typing import Dict, Any

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

    async def _query_(self, query: str, step: str | None):
        params = {
            "query": query,
            "step": step
        }
        base_url = "http://localhost:8428/prometheus"
        async with (aiohttp.ClientSession() as session):
            async with session.get(f"{base_url}/api/v1/query_range", params=params) as response:
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

    async def _query_libvirt(self, *,
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

    async def get_cpu_metrics_node_1m(self, uuid: str) -> dict[str, float | datetime]:
        query = (f'rate(libvirt_domain_info_cpu_time_seconds_total{{uuid="{uuid}"}}[1m]) * 100 / '
                 f'libvirt_domain_info_virtual_cpus{{uuid="{uuid}"}}')
        return await self._query_libvirt(query=query)

    async def get_memory_metrics_node_1m(self, uuid: str) -> dict[str, float | datetime]:
        memory_query = f'rate(libvirt_domain_memory_stats_used_percent{{uuid="{uuid}"}}[1m])'
        return await self._query_libvirt(query=memory_query)

    async def get_cpu_metrics_1m(self):
        cpu_utilization_query = ('sum by (instance) (rate(node_cpu_seconds_total{'
                                 'mode=~"user|system|iowait|irq|softirq|steal"}[1m])) * 100')
        return await self._query_(cpu_utilization_query, step="1m")

    async def get_memory_metrics_1m(self):
        memory_utilization_query = '100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)'
        return await self._query_(memory_utilization_query, step="1m")

    async def get_count_pod_metric(self) -> dict[str, Any]:
        node_count_query = 'count(kube_node_status_condition{condition="Ready", status="true"})'
        return await self._query_(query=node_count_query, step=None)

    async def get_cpu_memory_metrics_node_1m(self, uuid: str) -> dict:
        cpu = await self.get_cpu_metrics_node_1m(uuid)
        memory = await self.get_memory_metrics_node_1m(uuid)
        logger.info("metric fetched for: " + uuid)
        return {
            "timestamp": cpu["timestamp"],
            "cpu": cpu["value"],
            "memory": memory["value"],
        }

    async def get_cpu_memory_metrics_1m(self, ) -> dict:
        # cpu = await self.get_cpu_metrics_libvirt_for_node_1m(uuid)
        # memory = await self.get_memory_metrics_libvirt_for_node_1m(uuid)
        pod_count = await self.get_count_nodes_metric()
        cpu = await self.get_cpu_metrics_1m()
        memory = await self.get_memory_metrics_1m()
        logger.info("metric fetched ")
        return {
            "timestamp": cpu["timestamp"],
            "cpu": cpu["value"],
            "memory": memory["value"],
            "pod_count": pod_count,
        }

    # async def get_cpu_metrics_app(self, app: str) -> dict[str, float | datetime]:
    #     query = f'avg(rate(libvirt_domain_info_cpu_time_seconds_total{{app="{app}"}}[1m]) * 100 / libvirt_domain_info_virtual_cpus{{app="{app}"}})'
    #     return await self._query(query=query)

    # async def get_memory_metrics_app(self, app: str) -> dict[str, float | datetime]:
    #     memory_query = f'rate(libvirt_domain_memory_stats_used_percent{{app="{app}"}}[1m])'
    #     return await self._query(query=memory_query)

    # async def get_cpu_memory_metrics_app(self, app: str) -> dict:
    #     cpu = await self.get_cpu_metrics_app(app)
    #     memory = await self.get_memory_metrics_app(app)
    #     logger.info("metric fetched for: " + app)
    #     return {
    #         "timestamp": cpu["timestamp"],
    #         "cpu": cpu["value"],
    #         "memory": memory["value"],
    #     }
