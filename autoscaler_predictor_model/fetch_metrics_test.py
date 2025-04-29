import asyncio
from datetime import datetime

import aiohttp


async def _query_(query: str,
                  step: str = "1m",
                  ):
    params = {
        "query": query,
        "step": step
    }
    base_url = "http://localhost:8428/prometheus"
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/api/v1/query_range", params=params) as response:
            response.raise_for_status()
            data = await response.json()

            if data["status"] != "success":
                raise ValueError(f"Query failed: {data.get('error', 'Unknown error')}")

            return {
                "timestamp": data["data"]["result"][0]['values'][0][0],
                "value": data["data"]["result"][0]['values'][0][1]
            }
            # start_value = data["data"]["result"][0]["values"][0][1]
            # end_value = data["data"]["result"][0]["values"][1][1]
            # avg_value = (float(start_value) + float(end_value)) / 2
            # timestamp = datetime.fromtimestamp(int(data["data"]["result"][0]["values"][1][0]))  # Время конца минуты
            # return {
            #     "timestamp": timestamp,
            #     "value": avg_value
            # }


async def main():
    node_count_query = 'count(kube_pod_status_ready{condition="true",pod=~"simple-app-.*"})[1m]'
    result = await _query_(node_count_query)
    print("Полученный результат:", result)


data = asyncio.run(main())

print(data)
