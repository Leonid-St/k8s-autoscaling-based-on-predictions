from datetime import datetime, timedelta, timezone
from utils.prometheus_util import PrometheusUtil
from enum import Enum
import os
prom_url = "http://193.61.36.69:31000/"
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd

plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams["font.size"] = "10"
plt.rcParams['axes.xmargin'] = 0


class Queries(Enum):
    RPS = "rps_query"
    NUM_PODS = "num_pods_query"
    RESPONSE_TIME = "response_time_query"
    CPU_USAGE = "pod_cpu_usage_query"

class AutoScalers(Enum):
    KHPA = "khpa"
    NO_AUTO_SCALING = "no_autoscaling"
    PROACTIVE = "proactive"
    REACTIVE = "reactive"

class ExperimentalRuns(Enum):
    CONSTGROW_CPU = "ConstGrow,CPU,pythonwebapp2"
    CONSTGROW_RPS = "ConstGrow,RPS,pythonwebapp2"
    SPIKY_CPU = "Spiky,CPU,pythonwebapp2"
    SPIKY_RPS = "Spiky,RPS,pythonwebapp2"
    ZIGZAG_CPU = "ZigZag,CPU,pythonwebapp"
    ZIGZAG_RPS = "ZigZag,RPS,pythonwebapp"

class ExperimentRun:
    queries = {
        Queries.RPS: "(sum(rate(envoy_cluster_upstream_rq{envoy_cluster_name=\"$webapp\"}[1m])) + sum(rate(envoy_cluster_upstream_rq{envoy_cluster_name=\"$webapp\"}[1m] offset 30m)) + sum(rate(envoy_cluster_upstream_rq{envoy_cluster_name=\"$webapp\"}[1m] offset 60m)) + sum(rate(envoy_cluster_upstream_rq{envoy_cluster_name=\"$webapp\"}[1m] offset 90m)) + sum(rate(envoy_cluster_upstream_rq{envoy_cluster_name=\"$webapp\"}[1m] offset 120m)) + sum(rate(envoy_cluster_upstream_rq{envoy_cluster_name=\"$webapp\"}[1m] offset 150m)) ) / 6",
        Queries.NUM_PODS: "(count(count by (pod) (eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" })) +count(count by (pod) (eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" } offset 30m)) +count(count by (pod) (eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" } offset 60m)) +count(count by (pod) (eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" } offset 90m)) +count(count by (pod) (eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" } offset 120m)) +count(count by (pod) (eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" } offset 150m))) / 6",
        Queries.RESPONSE_TIME: "(histogram_quantile(0.95, rate( envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name=\"$webapp\"}[1m])) +histogram_quantile(0.95, rate( envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name=\"$webapp\"}[1m] offset 30m)) +histogram_quantile(0.95, rate( envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name=\"$webapp\"}[1m] offset 60m)) +histogram_quantile(0.95, rate( envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name=\"$webapp\"}[1m] offset 90m)) +histogram_quantile(0.95, rate( envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name=\"$webapp\"}[1m] offset 120m)) +histogram_quantile(0.95, rate( envoy_cluster_upstream_rq_time_bucket{envoy_cluster_name=\"$webapp\"}[1m] offset 150m))) / 6",
        Queries.CPU_USAGE: "(sum(eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" })/ sum(eagle_pod_container_resource_requests_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\"}) +sum(eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" } offset 30m)/ sum(eagle_pod_container_resource_requests_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\"} offset 30m) +sum(eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" } offset 60m)/ sum(eagle_pod_container_resource_requests_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\"} offset 60m) +sum(eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" } offset 90m)/ sum(eagle_pod_container_resource_requests_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\"} offset 90m) +sum(eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" } offset 120m)/ sum(eagle_pod_container_resource_requests_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\"} offset 120m) +sum(eagle_pod_container_resource_usage_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\" } offset 150m)/ sum(eagle_pod_container_resource_requests_cpu_cores{ pod=~\"$webapp-.*\", container!=\"POD\", container=~\"$webapp\", phase=\"Running\"} offset 150m))/6"
    }

    def __init__(self, type_of_run: ExperimentalRuns, webapp_type, khpa_cycle_start_time: datetime, offset_no_auto_scaling: int,
                 offset_reactive_scaling: int, offset_proactive_scaling: int):
        self.experiment_run_type = type_of_run
        self.webapp_type = webapp_type
        self.khpa_cycle_start_time = khpa_cycle_start_time
        self.offset_no_auto_scaling = offset_no_auto_scaling
        self.offset_proactive_scaling = offset_proactive_scaling
        self.offset_reactive_scaling = offset_reactive_scaling

    def __formatted_promql_queries(self):
        return {k: v.replace("$webapp", self.webapp_type) for k, v in self.queries.items()}


    def __get_folder_path(self, autoscaler_type: AutoScalers, query_type: Queries):
        return os.path.join("./data/", self.experiment_run_type.value, autoscaler_type.value, query_type.value)

    def __get_file_path(self, folder_path, file_name):
        return os.path.join(folder_path, file_name)


    def __get_all_metrics(self, autoscaler_type: AutoScalers, start_time:datetime, end_time:datetime):
        result = {}
        for (query_name, query) in self.__formatted_promql_queries().items():
            folder_path  = self.__get_folder_path(autoscaler_type=autoscaler_type, query_type=query_name)
            file_path = self.__get_file_path(folder_path, "data.csv")
            if (os.path.isfile(file_path) and os.access(file_path, os.R_OK)):
                print("Read cached data from {}".format(file_path))
                result[query_name] = pd.read_csv(file_path, index_col=0, squeeze=True, infer_datetime_format=True, parse_dates=[0])
            else:
                result_series = PrometheusUtil.app_metrics_query_helper(url=prom_url, query=query, metric_name="",
                                                                        pods=[], start_time=start_time, end_time=end_time, step=10,
                                                                        per_pod=False).metrics
                result[query_name] = result_series
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)
                print("Caching data into {}".format(file_path))
                pd.DataFrame(result_series).to_csv(file_path, header=True)
        return result

    def get_data(self):
        metrics = {}

        no_autoscaling_start_time = self.khpa_cycle_start_time - timedelta(minutes=self.offset_no_auto_scaling)
        metrics[AutoScalers.NO_AUTO_SCALING] = self.__get_all_metrics(AutoScalers.NO_AUTO_SCALING,
                                        no_autoscaling_start_time, no_autoscaling_start_time + timedelta(minutes=30))

        proactive_start_time = self.khpa_cycle_start_time - timedelta(minutes=self.offset_proactive_scaling)
        metrics[AutoScalers.PROACTIVE] = self.__get_all_metrics(AutoScalers.PROACTIVE,
                                        proactive_start_time, proactive_start_time + timedelta(minutes=30))

        reactive_start_time = self.khpa_cycle_start_time - timedelta(minutes=self.offset_reactive_scaling)
        metrics[AutoScalers.REACTIVE] = self.__get_all_metrics(AutoScalers.REACTIVE,
                                        reactive_start_time, reactive_start_time + timedelta(minutes=30))

        khpa_start_time = self.khpa_cycle_start_time
        metrics[AutoScalers.KHPA] = self.__get_all_metrics(AutoScalers.KHPA, khpa_start_time,
                                                           khpa_start_time + timedelta(minutes=30))

        return metrics

    def get_data_flat(self):
        metrics = self.get_data()
        noas_num_pods = metrics[AutoScalers.NO_AUTO_SCALING][Queries.NUM_PODS]
        noas_rps = metrics[AutoScalers.NO_AUTO_SCALING][Queries.RPS]
        noas_latency = metrics[AutoScalers.NO_AUTO_SCALING][Queries.RESPONSE_TIME]
        noas_cpu_usage = metrics[AutoScalers.NO_AUTO_SCALING][Queries.CPU_USAGE]

        khpa_num_pods = metrics[AutoScalers.KHPA][Queries.NUM_PODS]
        khpa_rps = metrics[AutoScalers.KHPA][Queries.RPS]
        khpa_latency = metrics[AutoScalers.KHPA][Queries.RESPONSE_TIME]
        khpa_cpu_usage = metrics[AutoScalers.KHPA][Queries.CPU_USAGE]

        react_num_pods = metrics[AutoScalers.REACTIVE][Queries.NUM_PODS]
        react_rps = metrics[AutoScalers.REACTIVE][Queries.RPS]
        react_latency = metrics[AutoScalers.REACTIVE][Queries.RESPONSE_TIME]
        react_cpu_usage = metrics[AutoScalers.REACTIVE][Queries.CPU_USAGE]

        proact_num_pods = metrics[AutoScalers.PROACTIVE][Queries.NUM_PODS]
        proact_rps = metrics[AutoScalers.PROACTIVE][Queries.RPS]
        proact_latency = metrics[AutoScalers.PROACTIVE][Queries.RESPONSE_TIME]
        proact_cpu_usage = metrics[AutoScalers.PROACTIVE][Queries.CPU_USAGE]

        return noas_num_pods, noas_rps, noas_latency, noas_cpu_usage, khpa_num_pods, khpa_rps, khpa_latency, khpa_cpu_usage, react_num_pods, react_rps, react_latency, react_cpu_usage, proact_num_pods, proact_rps, proact_latency, proact_cpu_usage


# to be manually tuned based on when the experiments were ran
# note the experiments don't have to be in a specific order, but correct offsets need to be given
experiments = {
    ExperimentalRuns.CONSTGROW_CPU: ExperimentRun(type_of_run=ExperimentalRuns.CONSTGROW_CPU, webapp_type="pythonwebapp2",
                  khpa_cycle_start_time=datetime(year=2020, month=8, day=22, hour=18, minute=0, second=0, tzinfo=timezone.utc),
                  offset_no_auto_scaling=630,
                  offset_proactive_scaling=420,
                  offset_reactive_scaling=210
                  ),
    ExperimentalRuns.CONSTGROW_RPS: ExperimentRun(type_of_run=ExperimentalRuns.CONSTGROW_RPS, webapp_type="pythonwebapp2",
                  khpa_cycle_start_time=datetime(year=2020, month=8, day=23, hour=14, minute=00, second=0, tzinfo=timezone.utc),
                  offset_no_auto_scaling=660,
                  offset_proactive_scaling=450,
                  offset_reactive_scaling=240
                  ),
    ExperimentalRuns.SPIKY_CPU: ExperimentRun(type_of_run=ExperimentalRuns.SPIKY_CPU, webapp_type="pythonwebapp2",
                  khpa_cycle_start_time=datetime(year=2020, month=8, day=26, hour=8, minute=30, second=0, tzinfo=timezone.utc),
                  offset_no_auto_scaling=630,
                  offset_proactive_scaling=420,
                  offset_reactive_scaling=210
                  ),
    ExperimentalRuns.SPIKY_RPS: ExperimentRun(type_of_run=ExperimentalRuns.SPIKY_RPS, webapp_type="pythonwebapp2",
                  khpa_cycle_start_time=datetime(year=2020, month=9, day=9, hour=17, minute=0, second=0, tzinfo=timezone.utc),
                  offset_no_auto_scaling=22980,
                  offset_proactive_scaling=22680,
                  offset_reactive_scaling=22110
                  ),
    ExperimentalRuns.ZIGZAG_CPU: ExperimentRun(type_of_run=ExperimentalRuns.ZIGZAG_CPU, webapp_type="pythonwebapp",
                  khpa_cycle_start_time=datetime(year=2020, month=8, day=25, hour=12, minute=0, second=0, tzinfo=timezone.utc),
                  offset_no_auto_scaling=1080,
                  offset_proactive_scaling=870,
                  offset_reactive_scaling=210
                  ),
    ExperimentalRuns.ZIGZAG_RPS: ExperimentRun(type_of_run=ExperimentalRuns.ZIGZAG_RPS, webapp_type="pythonwebapp",
                  khpa_cycle_start_time=datetime(year=2020, month=8, day=24, hour=9, minute=30, second=0, tzinfo=timezone.utc),
                  offset_no_auto_scaling=900,
                  offset_proactive_scaling=690,
                  offset_reactive_scaling=480
                  )
}


def plot_num_pods_rps(myax, index, label, num_pods, rps):
    myaxb = myax.twinx()

    myax.text(0.5, 1.25, label, size=16, fontweight='bold', ha="center", transform=myax.transAxes)

    # Same as above
    myax.set_xlabel('Time (minutes)', fontsize=14)
    myax.set_ylabel('Requests per second', fontsize=14)
    myax.set_title(index, fontweight="bold", fontsize=14)
    myax.grid(True)

    # myax.fill_between(num_pods.index, num_pods.values, color='tab:green', step="pre", alpha=0.2)
    # Plotting on the first y-axis
    rps.index = (pd.Timestamp('20200101') + (rps.index - rps.index.values[0]))
    myax.plot(rps.index, rps.values, color='tab:green', label='Requests per second')

    # Plotting on the second y-axis
    myaxb.set_ylabel('Number of running pods', fontsize=14)
    # myaxb.fill_between(rps.index, rps.values, color='tab:orange', step="pre", alpha=0.2)
    num_pods.index = (pd.Timestamp('20200101') + (num_pods.index - num_pods.index.values[0]))
    myaxb.plot(num_pods.index, num_pods.values, color='tab:red', label='Number of running pods')

    myax.xaxis.set_major_formatter(DateFormatter('%M:%S'))

    # Handling of getting lines and labels from all axes for a single legend
    mylines, mylabels = myax.get_legend_handles_labels()
    mylines2, mylabels2 = myaxb.get_legend_handles_labels()
    myax.legend(mylines + mylines2, mylabels + mylabels2, loc='upper left', fontsize=10)


def plot_response_time(myax, index, response_time):
    myax.set_xlabel('Time (minutes)', fontsize=14)
    myax.set_ylabel('95th percentile response time (ms)', fontsize=14)
    myax.set_title(index, fontweight="bold", fontsize=14)
    myax.grid(True)
    # response_time.fillna(-0.5, inplace=True)
    # myax.fill_between(response_time.index, response_time.values, step="pre", alpha=0.2)
    response_time.index = (pd.Timestamp('20200101') + (response_time.index - response_time.index.values[0]))
    myax.plot(response_time.index, response_time.values, color='tab:green')
    myax.set_xlim([min(response_time.index.values), max(response_time.index.values)])
    # myax.set_ylim(bottom=0)
    myax.xaxis.set_major_formatter(DateFormatter('%M:%S'))


def plot_cpu_usage(myax, index, cpu_usage):
    myax.set_xlabel('Time (minutes)', fontsize=14)
    myax.set_ylabel('Average pod CPU usage (%)', fontsize=14)
    myax.set_ylim([0, 100])
    myax.set_title(index, fontweight="bold", fontsize=14)
    myax.grid(True)
    # myax.fill_between(cpu_usage.index, cpu_usage.values, step="pre", alpha=0.2)
    cpu_usage.index = (pd.Timestamp('20200101') + (cpu_usage.index - cpu_usage.index.values[0]))
    myax.plot(cpu_usage.index, cpu_usage.values * 100, color='tab:green')
    myax.xaxis.set_major_formatter(DateFormatter('%M:%S'))



def main(exp_name: ExperimentalRuns):
    experimental_run = experiments[exp_name]
    noas_num_pods, noas_rps, noas_latency, noas_cpu_usage, khpa_num_pods, khpa_rps, khpa_latency, khpa_cpu_usage, \
    react_num_pods, react_rps, react_latency, react_cpu_usage, \
    proact_num_pods, proact_rps, proact_latency, proact_cpu_usage = experimental_run.get_data_flat()

    fig, axs = plt.subplots(ncols=4, nrows=3, gridspec_kw={"height_ratios": [8, 8, 8], "wspace": 0.4, "hspace": 0.33})

    plot_num_pods_rps(axs[0,0], "1(a)", "1) No auto-scaling", noas_num_pods, noas_rps)
    plot_num_pods_rps(axs[0,1], "2(a)", "2) Kubernetes horizontal pod auto-scaler", khpa_num_pods, khpa_rps)
    plot_num_pods_rps(axs[0,2], "3(a)", "3) Reactive auto-scaler", react_num_pods, react_rps)
    plot_num_pods_rps(axs[0,3], "4(a)", "4) KubeScale auto-scaler", proact_num_pods, proact_rps)
    plot_response_time(axs[1,0], "1(b)", noas_latency)
    plot_response_time(axs[1,1], "2(b)", khpa_latency)
    plot_response_time(axs[1,2], "3(b)", react_latency)
    plot_response_time(axs[1,3], "4(b)", proact_latency)
    plot_cpu_usage(axs[2,0], "1(c)", noas_cpu_usage)
    plot_cpu_usage(axs[2,1], "2(c)", khpa_cpu_usage)
    plot_cpu_usage(axs[2,2], "3(c)", react_cpu_usage)
    plot_cpu_usage(axs[2,3], "4(c)", proact_cpu_usage)

    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.90)

    plt.savefig(os.path.join(".", "experiment_runs", exp_name.value), dpi=200, overwrite=True)


if __name__ == '__main__':
    main(exp_name=ExperimentalRuns.CONSTGROW_CPU)
    main(exp_name=ExperimentalRuns.CONSTGROW_RPS)

    main(exp_name=ExperimentalRuns.ZIGZAG_CPU)
    main(exp_name=ExperimentalRuns.ZIGZAG_RPS)

    main(exp_name=ExperimentalRuns.SPIKY_CPU)
    main(exp_name=ExperimentalRuns.SPIKY_RPS)
