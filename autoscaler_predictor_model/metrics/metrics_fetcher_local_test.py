from datetime import datetime
from metrics.metrics_fetcher import MetricsFetcher
import asyncio
import random


class MetricsFetcherLocalTest(MetricsFetcher):
    async def get_cpu_metrics_1m(self, uuid: str) -> dict[str, float | datetime]:
        await asyncio.sleep(0.1)  #  Имитация задержки запроса
        return {
            "timestamp": datetime.now(),
            "value": random.uniform(10, 90)  #  Случайное значение CPU от 10% до 90%
        }

    async def get_memory_metrics_1m(self, uuid: str) -> dict[str, float | datetime]:
        await asyncio.sleep(0.1)  #  Имитация задержки запроса
        return {
            "timestamp": datetime.now(),
            "value": random.uniform(20, 80)  #  Случайное значение memory от 20% до 80%
        }

    async def get_cpu_memory_metrics_1m(self, uuid: str) -> dict:
        cpu = await self.get_cpu_metrics_1m(uuid)
        memory = await self.get_memory_metrics_1m(uuid)
        return {
            "timestamp": cpu["timestamp"],
            "cpu": cpu["value"],
            "memory": memory["value"],
        } 
