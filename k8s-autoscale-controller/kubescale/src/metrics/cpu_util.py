from utils.kubernetes_util import PodWrapper
from utils.prometheus_util import PrometheusUtil, MetricWrapper
from typing import List, Dict, Optional, Union
import datetime
from metrics.metric_src import MetricSource


class CpuUsageTotal(MetricSource):
    def __init__(self, prom_url: str, step_size_secs: int, deployment_name: str):
        self.metric_name = "cpu_usage_per_pod"
        self.pod_name = deployment_name + "-.*"
        self.query_template = """
            sum(eagle_pod_container_resource_usage_cpu_cores{{ pod=~"{pod_name}", container!="POD", container=~"{container_name}", phase="Running"}})          
            / (
                sum(eagle_pod_container_resource_requests_cpu_cores{{ pod=~"{pod_name}", container!="POD", container=~"{container_name}", phase="Running" }}) / 
                count(count by (pod) (eagle_pod_container_resource_usage_cpu_cores{{ pod=~"{pod_name}", container!="POD", container=~"{container_name}", phase="Running" }}))
            )
        """
        self.prom_url: str = prom_url
        self.step_size_secs: int = step_size_secs
        self.aggregate_per_pod: bool = False

    def get_data(self, pods: List[PodWrapper], start_time: datetime.datetime, end_time: datetime.datetime) \
            -> Optional[Union[MetricWrapper, Dict[PodWrapper,MetricWrapper]]]:
        if len(pods) == 0:
            return None
        else:
            query = self.query_template.format(
                pod_name=self.pod_name, container_name=pods[0].container_name
            )
            return PrometheusUtil.app_metrics_query_helper(url=self.prom_url, query=query,
                                                           metric_name=self.metric_name, pods=pods,
                                                           start_time=start_time, end_time=end_time,
                                                           step=self.step_size_secs,
                                                           per_pod=self.aggregate_per_pod)
