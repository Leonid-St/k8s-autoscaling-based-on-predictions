import requests
import datetime
import json
from typing import List, Dict, Union
from .kubernetes_util import PodWrapper
import pandas as pd
import logging


class MetricWrapper():
    def __init__(self, metric_name: str, metrics: pd.Series):
        self.metric_name: str = metric_name
        self.metrics: pd.Series = metrics

class PrometheusUtil:
    @staticmethod
    def query_time_range(url: str, query: str, start_time: datetime.datetime, end_time: datetime.datetime,
                         step: float):
        prefix = "/api/v1/query_range"
        """
        :param query: the query to be submitted to prometheus - you must include the grouping in the query
        :param start_time: eg. datetime.datetime = datetime.datetime(year=2020, month=1, day=24, hour=1, minute=0, second=0, tzinfo=datetime.timezone.utc)
        :param end_time: end time eg. datetime.datetime = datetime.datetime(year=2020, month=1, day=24, hour=1, minute=0, second=0, tzinfo=datetime.timezone.utc)
        :param step: number of seconds between each value in the given time range
        :return: a dataframe indexed by time column and containing the metrics columns
        """
        payload = {
            'query': query,
            'start': int(start_time.timestamp()),
            'end': int(end_time.timestamp()),
            'step': step
        }
        response = requests.get(url + prefix, params=payload)
        response.raise_for_status()
        return json.loads(response.content.decode("utf-8"))["data"]["result"]

    """
    :arg aggregate_per_pod -> specifies that metrics are returned for each pod instead of being aggregated
    :arg by_keyword -> specifies how to group the metrics
    """
    @staticmethod
    def app_metrics_query_helper(url: str, query:str, metric_name:str, pods: List[PodWrapper],
                                 start_time: datetime.datetime, end_time: datetime.datetime,
                                 step: float, per_pod=False, by_keyword:str = "") -> Union[Dict[PodWrapper,
                                                                                                MetricWrapper], MetricWrapper]:
        # queries prometheus using PromQL queries
        # returns a dataframe
        logging.debug("Executing query {}".format(query))
        data: List[Dict] = PrometheusUtil.query_time_range(url=url, query=query, start_time=start_time,
                                                           end_time=end_time, step=step)
        # unlikely that this will be used, not too useful overall
        if per_pod:
            # now tie back the data to PodWrapper and return the MetricWrapper objects
            results: Dict[PodWrapper, MetricWrapper] = {}
            for i in data:
                matching_pod = None
                for pod in pods:
                    if pod.prometheus_ip in i["metric"][by_keyword] or pod._pod.metadata._name in i["metric"][by_keyword]:
                        matching_pod = pod
                if matching_pod is None:
                    raise ValueError("no matching pods...")
                #
                ts_data = pd.Series(data=[float(i[1]) for i in data[0]["values"]],
                                    index=[datetime.datetime.utcfromtimestamp(i[0]) for i in data[0]["values"]])
                results[matching_pod] = MetricWrapper(metric_name=metric_name, metrics=ts_data)
            return results
        else:
            ts_data = pd.Series(data=[float(i[1]) for i in data[0]["values"]],
                                index=[datetime.datetime.utcfromtimestamp(i[0]) for i in data[0]["values"]])
            return MetricWrapper(metric_name=metric_name, metrics=ts_data)

    @staticmethod
    def merge_metrics(pods: List[PodWrapper], metrics: List[Dict[PodWrapper, MetricWrapper]]):
        merged_metrics = {}
        for pod in pods:
            res: List[MetricWrapper] = []
            for i in range(0, len(metrics)):
                res.append(metrics[i][pod])
            merged_df = pd.DataFrame({k.metric_name: k.metrics for k in res})
            merged_metrics[pod] = merged_df
        return merged_metrics

    # # NOTE: these queries are highly specific to the current application setup
    # def query_requests_per_second(self, pods: List[PodWrapper], start_time: datetime.datetime,
    #                               end_time: datetime.datetime, step: float) -> Dict[PodWrapper, MetricWrapper]:
    #     metric_name = "app_requests_per_second_per_instance"
    #     pods_string = ".*.|".join([pod.prometheus_ip for pod in pods]) + ".*."
    #     by_keyword = "instance"
    #     endpoint = "/bmark/request/{id}"
    #     query = """ sum(irate(http_server_requests_seconds_count{{
    #     instance=~"{pod_name}", uri="{endpoint}"}}[1m])) by ({by_keyword})"""\
    #         .format(pod_name=pods_string, by_keyword=by_keyword, endpoint=endpoint)
    #     return self.app_metrics_query_helper(query, by_keyword, metric_name, pods, start_time, end_time, step)
    #
    # # NOTE: these queries are highly specific to the current application setup
    # def query_latency(self, pods: List[PodWrapper], start_time: datetime.datetime, end_time: datetime.datetime,
    #                   step: float) -> Dict[PodWrapper, MetricWrapper]:
    #     metric_name = "app_request_latency_per_instance"
    #     pods_string = ".*.|".join([pod.prometheus_ip for pod in pods]) + ".*."
    #     by_keyword = "instance"
    #     endpoint = "/bmark/request/{id}"
    #     query = """ sum(
    #         rate(http_server_requests_seconds_sum{{instance=~"{pod_name}", uri="{endpoint}"}}[1m]) /
    #         rate(http_server_requests_seconds_count{{instance=~"{pod_name}", uri="{endpoint}"}}[1m]))
    #         by ({by_keyword})""".format(pod_name=pods_string, endpoint=endpoint, by_keyword=by_keyword)
    #     return self.app_metrics_query_helper(query, by_keyword, metric_name, pods, start_time, end_time, step)
    #
    # # NOTE: these queries are highly specific to the current application setup
    # def query_cpu_utilisation(self, pods: List[PodWrapper], start_time: datetime.datetime, end_time: datetime.datetime,
    #                           step: float) -> Dict[PodWrapper, MetricWrapper]:
    #     metric_name = "cpu_usage_per_pod"
    #     pods_string = "|".join([pod._pod.metadata._name for pod in pods])
    #     container_name = pods[0].container_name
    #     by_keyword = "pod"
    #     query = """
    #         sum(eagle_pod_container_resource_usage_cpu_cores{{ pod=~"{pods}", container!="POD", container=~"{container_name}" }}) by ({by_keyword})
    #         / sum(eagle_pod_container_resource_limits_cpu_cores{{ pod=~"{pods}", container!="POD", container=~"{container_name}" }}) by ({by_keyword})""".format(
    #         pods=pods_string, container_name=container_name, by_keyword=by_keyword
    #     )
    #     return self.app_metrics_query_helper(query, by_keyword, metric_name, pods, start_time, end_time, step)
    # # NOTE: these queries are highly specific to the current application setup
    # def query_memory_utilisation(self, pods: List[PodWrapper], start_time: datetime.datetime,
    #                              end_time: datetime.datetime, step: float) -> Dict[PodWrapper, MetricWrapper]:
    #     metric_name = "memory_usage_per_pod"
    #     pods_string = "|".join([pod._pod.metadata._name for pod in pods])
    #     container_name = pods[0].container_name
    #     by_keyword = "pod"
    #     query = """
    #         sum(eagle_pod_container_resource_usage_memory_bytes{{ pod=~"{pods}", container!="POD", container=~"{container_name}" }}) by ({by_keyword})
    #         / sum(eagle_pod_container_resource_limits_memory_bytes{{ pod=~"{pods}", container!="POD", container=~"{container_name}" }}) by ({by_keyword})""".format(
    #         pods=pods_string, container_name=container_name, by_keyword=by_keyword
    #     )
    #     return self.app_metrics_query_helper(query, by_keyword, metric_name, pods, start_time, end_time, step)
