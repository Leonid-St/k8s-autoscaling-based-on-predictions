from kubernetes import config
from utils.kubernetes_util import KubernetesUtil, PodWrapper
from utils.prometheus_util import MetricWrapper
from utils.controller_utils import *
import time
from math import ceil, floor
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import os
import yaml
from yaml import FullLoader
from metrics.metric_src import MetricSource
from utils.ml_utils import generate_forecast, build_model
from gluonts.model.forecast import Forecast
import logging
logging.basicConfig(
    format='%(levelname)s: %(asctime)s %(message)s',
    level=logging.INFO
)
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import atexit
registry = CollectorRegistry()
prom_gauge = Gauge('scaling_mode', '0 = reactive mode, 1 = proactive mode', ["scaler_location", "deployment"],
                   registry=registry)
deployment_name = None
pushgateway_url = None


def check_running_in_docker():
    path = '/proc/self/cgroup'
    return (os.path.isfile(path) and any('docker' in l for l in open(path, 'r'))) or os.path.exists('/.dockerenv')


def exit_handler():
    deployed = "k8s" if check_running_in_docker() else "local"
    prom_gauge.labels(scaler_location=deployed, deployment=deployment_name).set(int(-1))
    push_to_gateway(pushgateway_url, job="auto-scaler-{}-{}".format(deployed, deployment_name), registry=registry)


class Controller:
    def __init__(self, config_path: str):
        self.yaml_content = yaml.load(open(config_path, 'r'), Loader=FullLoader)
        logger.info("Yaml content is {}".format(self.yaml_content))
        self.prometheus_namespace = self.yaml_content["prometheus_namespace"]
        self.prometheus_svc_name = self.yaml_content["prometheus_svc_name"]
        self.prometheus_url = "http://{ip}:{port}".format(**KubernetesUtil.get_service_ip_and_port(
            namespace_name=self.prometheus_namespace, service_name=self.prometheus_svc_name))

        self.pushgateway_namespace = self.yaml_content["pushgateway_namespace"]
        self.pushgateway_svc_name = self.yaml_content["pushgateway_svc_name"]
        self.pushgateway_url = "http://{ip}:{port}".format(**KubernetesUtil.get_service_ip_and_port(
                namespace_name=self.pushgateway_namespace, service_name=self.pushgateway_svc_name))
        global pushgateway_url
        pushgateway_url  = self.pushgateway_url

        self.notification_email_enabled = self.yaml_content.get("notification_email_enabled", None)
        self.notification_email_receiver = self.yaml_content.get("notification_email_receiver", None)
        if self.notification_email_enabled is not None and self.notification_email_enabled == False:
            self.notification_email_receiver = None
        elif self.notification_email_enabled == True and self.notification_email_receiver == None:
            logger.error("Please enter a valid email for the email receiver")

        self.namespace = self.yaml_content["kubernetes"]["namespace"]

        global deployment_name
        self.deployment = self.yaml_content["kubernetes"]["deployment"]
        deployment_name = self.deployment
        self.container = self.yaml_content["kubernetes"]["container"]

        self.reactive_scaling_enabled = self.yaml_content["strategy"]["reactive_scaling_enabled"]
        self.proactive_scaling_enabled = self.yaml_content["strategy"]["proactive_scaling_enabled"]
        self.proactive_downscaling_enabled = self.yaml_content["strategy"]["proactive_downscaling_enabled"]
        self.delay_proactive_mode_by_min_train_data_history = self.yaml_content["strategy"]["delay_proactive_mode_by_min_train_data_history"]
        self.proactive_mode_forecast_only = self.yaml_content["strategy"]["proactive_mode_forecast_only"]
        self.eval_time_interval_sec = self.yaml_content["strategy"]["eval_time_interval_sec"]
        if self.eval_time_interval_sec < 30:
            raise ValueError()
        self.min_instances = self.yaml_content["strategy"]["min_instances"]
        self.max_instances = self.yaml_content["strategy"]["max_instances"]
        self.downscale_cooldown_period_min = self.yaml_content["strategy"]["downscale_cooldown_period_min"]
        self.downscale_after_predictive_scaleup_cooldown_period_min = \
            self.yaml_content["strategy"]["downscale_after_predictive_scaleup_cooldown_period_min"]
        self.preempt_period_min = self.yaml_content["strategy"]["preempt_period_min"]
        self.downscale_max_percentage = self.yaml_content["strategy"].get("downscale_max_percentage")

        self.forecast_creation_interval_mins = self.yaml_content["forecasting"]["forecast_creation_interval_mins"]
        self.forecast_period_mins = self.yaml_content["forecasting"]["forecast_period_mins"]

        self.load_metric_name = self.yaml_content["metrics"]["load_metric_name"]
        self.scaling_metric_target_value = self.yaml_content["metrics"]["scaling_metric_target_value"]
        self.threshold_breach_tolerance = self.yaml_content["metrics"]["threshold_breach_tolerance"]
        self.min_train_data_history_hours = self.yaml_content["metrics"]["min_train_data_history_hours"]
        self.max_train_data_history_hours = self.yaml_content["metrics"]["max_train_data_history_hours"]
        self.step_size_mins = self.yaml_content["metrics"]["step_size_mins"]
        self.custom_params = self.yaml_content.get("custom_params",{})

        self.frequency = str(self.step_size_mins) + "min"
        self.latest_forecast: Optional[Forecast] = None
        self.scaling_decisions: List[ScalingDecision] = []

        if self.delay_proactive_mode_by_min_train_data_history and self.proactive_scaling_enabled:
            self.process_start_time = datetime.now(timezone.utc)
            self.proactive_scaling_enabled = False
            self.proactive_downscaling_enabled = False

        deployed = "k8s" if check_running_in_docker() else "local"
        prom_gauge.labels(scaler_location=deployed, deployment=deployment_name).set(int(self.proactive_scaling_enabled))
        push_to_gateway(pushgateway_url, job="auto-scaler-{}-{}".format(deployed, deployment_name), registry=registry)


    def get_metric_source(self) -> MetricSource:
        if self.load_metric_name == "cpu_usage":
            from src.metrics.cpu_util import CpuUsageTotal
            metric_source = CpuUsageTotal(prom_url=self.prometheus_url, step_size_secs=60*self.step_size_mins,
                                          deployment_name=self.deployment)
        elif self.load_metric_name == "requests_per_second":
            from src.metrics.requests_per_second import RequestsPerSecond
            metric_source = RequestsPerSecond(prom_url=self.prometheus_url, step_size_secs=60*self.step_size_mins,
                                              params=self.custom_params)
        else:
            raise ValueError("Wrong input metric source")
        return metric_source

    def check_min_train_data_history_time_passed(self, current_time: datetime):
        if (self.process_start_time < (current_time - timedelta(hours=self.min_train_data_history_hours,minutes=1))):
            self.delay_proactive_mode_by_min_train_data_history = False
            self.proactive_scaling_enabled = self.yaml_content["strategy"]["proactive_scaling_enabled"]
            self.proactive_downscaling_enabled = self.yaml_content["strategy"]["proactive_downscaling_enabled"]
            deployed = "k8s" if check_running_in_docker() else "local"
            prom_gauge.labels(scaler_location=deployed).set(int(self.proactive_scaling_enabled))
            push_to_gateway(pushgateway_url, job="auto-scaler-{}".format(deployed), registry=registry)

    def start(self):
        metric_source = self.get_metric_source()
        last_scaling_decision_time: datetime = datetime.now(timezone.utc) - timedelta(seconds=60000)
        last_forecast_time: datetime = datetime.now(timezone.utc) - timedelta(seconds=60000)

        while True:
            start_time = datetime.now()
            # there are two dates:
            # a) current time precise - this is the actual UTC time with second/microsecond precision
            #                         - used for anything related to the auto-scaler decision making
            # b) current time - this is the actual UTC time with second/microsecond precision cut off
            #                 - used for reporting and talking with prometheus
            current_time_precise: datetime = datetime.now(timezone.utc)
            current_time = current_time_precise - timedelta(seconds=current_time_precise.second,
                                                            microseconds=current_time_precise.microsecond)
            if self.delay_proactive_mode_by_min_train_data_history and self.proactive_scaling_enabled:
                self.check_min_train_data_history_time_passed(current_time=current_time)
            try:
                # get the current number of pods
                pods: List[PodWrapper] = KubernetesUtil.get_pods_for_deployment(
                    namespace=self.namespace, deployment=self.deployment, container=self.container)
                current_num_inst = len(pods)
                new_num_inst = current_num_inst
                # initialise variables and hold values for temporary variables
                metric_wrapper: Optional[MetricWrapper] = None
                current_metric_value: Optional[float] = None
                scaling_type: Optional[ScalingDecisionType] = None

                ############################
                # PROACTIVE COMPONENT
                ############################
                if self.proactive_scaling_enabled == True:
                    scaling_type = ScalingDecisionType.PROACTIVE
                    # we have a maximum training time to avoid bringing in irrelevant data
                    training_data_start_time = current_time - timedelta(hours=self.max_train_data_history_hours)
                    metric_wrapper = metric_source.get_data(
                        pods=pods, start_time=training_data_start_time, end_time=current_time_precise)
                    metric_wrapper.metrics = metric_wrapper.metrics.reindex(
                        pd.date_range(
                            metric_wrapper.metrics.index.min(), metric_wrapper.metrics.index.max(), freq="1min"),
                        method=None, copy=True)
                    current_metric_value = metric_wrapper.metrics[-1] / current_num_inst
                    train_data_min_timestamp = pd.to_datetime(metric_wrapper.metrics.index.min(), utc=True)

                    # if there is enough data -> minimum data timestamp is earlier than
                    # minimum training data time (ie. current time - min training data history)
                    # we will train a predictive model
                    if train_data_min_timestamp < (current_time - timedelta(hours=self.min_train_data_history_hours,
                                                                            minutes=1)):
                        # there is a buffer time of 2 minutes, so if last forecast time = 13:00 and forecast_creation_interval_mins = 60
                        # then if current time = 13:57 -> 13:57 - 0:58 = 12:59 -> which is less than last forecast time
                        # and if current time = 13:59 -> 13:59 - 0:58 = 13:01 -> which is more than last forecast time
                        if ((current_time - timedelta(minutes=self.forecast_creation_interval_mins - 2))
                                >= last_forecast_time):
                            last_forecast_time = current_time
                            self.predictive_model = build_model(metric_wrapper.metrics, self.frequency, self.forecast_period_mins)

                        self.latest_forecast = generate_forecast(self.predictive_model, metric_wrapper.metrics, self.frequency)
                        self.forecast_series: pd.Series = self.latest_forecast.mean_ts

                        logger.debug("Evaluating proactive scaling")
                        prediction_time: datetime = current_time + timedelta(minutes=self.preempt_period_min)
                        # take 0 if prediction is negative
                        load_metric_prediction: float = max(0, self.forecast_series[prediction_time])
                        # metric above deals with total load, not per instance load
                        predicted_metric_value: float = load_metric_prediction / current_num_inst

                        prediction_explanation = generate_predictive_metric_value_logging_text(
                                                    prediction_time=prediction_time,
                                                    current_time=current_time,
                                                    current_num_inst=current_num_inst,
                                                    current_metric_value=current_metric_value,
                                                    predicted_metric_value=predicted_metric_value)
                        logger.info(prediction_explanation)
                        # case when predicted value above the target value
                        if predicted_metric_value > (self.scaling_metric_target_value + self.threshold_breach_tolerance):
                            new_num_inst = min([
                                ceil(current_num_inst * (predicted_metric_value / self.scaling_metric_target_value)),
                                self.max_instances
                            ])
                            if new_num_inst != current_num_inst:
                                if self.proactive_mode_forecast_only == True:
                                    new_num_inst = current_num_inst
                                scaling_explanation = generate_scaling_logging_text(
                                    type=scaling_type, new_num_inst=new_num_inst,
                                    current_num_inst=current_num_inst, forecast_only=self.proactive_mode_forecast_only,
                                    current_time=current_time)
                                logger.info(scaling_explanation)

                                notify_scaling_decision(email_receiver=self.notification_email_receiver, time=current_time,
                                    data=metric_wrapper.metrics,
                                    deployment_name=self.deployment,
                                    latest_forecast=self.latest_forecast,
                                    metric_value_explanation=prediction_explanation,
                                    scaling_explanation=scaling_explanation,
                                    proactive=self.proactive_scaling_enabled)
                        # proactive downscaling doesn't really make sense since your application performance will likely
                        # immediately suffer due to the removal of resources that are currently used
                        elif self.proactive_downscaling_enabled and \
                                predicted_metric_value < (self.scaling_metric_target_value - self.threshold_breach_tolerance):
                            if floor((current_time_precise - last_scaling_decision_time).seconds/60) > self.downscale_cooldown_period_min:
                                max_comparators = [
                                    ceil(current_num_inst * (predicted_metric_value / self.scaling_metric_target_value)),
                                    self.min_instances
                                ]
                                # if we don't want to allow going from eg. 100 instances to 1 instance
                                if self.downscale_max_percentage is not None:
                                    max_comparators.append(ceil(current_num_inst * (self.downscale_max_percentage/100)))
                                new_num_inst = max(max_comparators)

                                if new_num_inst != current_num_inst:
                                    scaling_explanation = generate_scaling_logging_text(
                                        type=scaling_type, new_num_inst=new_num_inst,
                                        current_num_inst=current_num_inst,
                                        forecast_only=self.proactive_mode_forecast_only,
                                        current_time=current_time)
                                    logger.info(scaling_explanation)
                                    notify_scaling_decision(email_receiver=self.notification_email_receiver, time=current_time,
                                        data=metric_wrapper.metrics,
                                        deployment_name=self.deployment,
                                        latest_forecast=self.latest_forecast,
                                        metric_value_explanation=prediction_explanation,
                                        scaling_explanation=scaling_explanation,
                                        proactive=self.proactive_downscaling_enabled)
                                    # revert the new number of instances to current to ensure no scaling is done
                                    if self.proactive_mode_forecast_only == True:
                                        new_num_inst = current_num_inst
                            else:
                                logger.info("No further proactive downscaling due to cool-down period")
                    else:
                        logger.info("We can't do proactive scaling since there isn't enough data")

                ############################
                # REACTIVE COMPONENT
                ############################
                if self.reactive_scaling_enabled == True and new_num_inst == current_num_inst:
                    scaling_type = ScalingDecisionType.REACTIVE
                    logger.debug("Evaluating reactive scaling")
                    # difference to predictive is that we only care about the current value (found in last 15 minutes)
                    min_data_start_time = datetime.now(timezone.utc) - timedelta(minutes=60)
                    if metric_wrapper is None:
                        metric_wrapper = metric_source.get_data(pods=pods, start_time=min_data_start_time,
                                                                end_time=current_time_precise)
                    if current_metric_value is None:
                        current_metric_value = metric_wrapper.metrics[-1] / current_num_inst

                    reactive_explanation = generate_reactive_metric_value_logging_text(current_time=current_time,
                        current_num_inst=current_num_inst, current_metric_value=current_metric_value)
                    logger.info(reactive_explanation)
                    # case when current value above the target value
                    if current_metric_value > (self.scaling_metric_target_value + self.threshold_breach_tolerance):
                        new_num_inst = min([
                            ceil(current_num_inst * (current_metric_value / self.scaling_metric_target_value)),
                                 self.max_instances
                        ])
                        if new_num_inst != current_num_inst:
                            scaling_explanation = generate_scaling_logging_text(type=scaling_type,
                                new_num_inst=new_num_inst, current_num_inst=current_num_inst, current_time=current_time)
                            logger.info(scaling_explanation)
                            notify_scaling_decision(email_receiver=self.notification_email_receiver, time=current_time,
                                data=metric_wrapper.metrics,
                                deployment_name=self.deployment,
                                latest_forecast=self.latest_forecast,
                                metric_value_explanation=reactive_explanation,
                                scaling_explanation=scaling_explanation,
                                proactive=self.proactive_scaling_enabled)
                    # case when current value below the target value
                    elif current_metric_value < (self.scaling_metric_target_value - self.threshold_breach_tolerance):
                        # if the last scaling decision was a proactive scale-up, then increase the cooldown period to
                        # the specific cooldown period that goes after a proactive scale up. this is in order to avoid
                        # the case where current metric value at time T is 20 but the proactive scaler just predicted
                        # that it will be 100 at T + 2. In that case the reactive auto-scaler might at T+1 downscale
                        # the app back to the number of instances at time T that was sufficient for metric value of 20
                        if (len(self.scaling_decisions) > 0
                                and self.scaling_decisions[-1].direction == ScalingDecisionDirection.UP
                                and self.scaling_decisions[-1].type == ScalingDecisionType.PROACTIVE):
                            cooldown_period = self.downscale_after_predictive_scaleup_cooldown_period_min
                            logger.info("Changing cooldown period to {} minutes due to last decision being scale-up "
                                         "by proactive auto-scaler".format(
                                        self.downscale_after_predictive_scaleup_cooldown_period_min))
                        else:
                            cooldown_period = self.downscale_cooldown_period_min
                        if floor((current_time_precise - last_scaling_decision_time).seconds / 60) >= cooldown_period:
                            max_comparators = [
                                ceil(current_num_inst * (current_metric_value / self.scaling_metric_target_value)),
                                self.min_instances
                            ]
                            if self.downscale_max_percentage is not None:
                                max_comparators.append(ceil(current_num_inst * (self.downscale_max_percentage/100)))
                            new_num_inst = max(max_comparators)
                            if new_num_inst != current_num_inst:
                                scaling_explanation = generate_scaling_logging_text(
                                    type=scaling_type, new_num_inst=new_num_inst,
                                    current_num_inst=current_num_inst, current_time=current_time,
                                )
                                notify_scaling_decision(email_receiver=self.notification_email_receiver, time=current_time,
                                    data=metric_wrapper.metrics,
                                    deployment_name=self.deployment,
                                    latest_forecast=self.latest_forecast,
                                    metric_value_explanation=reactive_explanation,
                                    scaling_explanation=scaling_explanation,
                                    proactive=self.proactive_scaling_enabled)
                        else:
                            logger.info("No further reactive downscaling due to cool-down period of {} minutes".format(
                                cooldown_period))

                ############################
                # IMPLEMENT THE SCALING DECISION
                ############################
                if new_num_inst != current_num_inst:
                    KubernetesUtil.set_number_of_replicas(deployment=self.deployment, namespace=self.namespace,
                                                          replicas=new_num_inst)
                    if new_num_inst > current_num_inst:
                        scaling_direction = ScalingDecisionDirection.UP
                    else:
                        scaling_direction = ScalingDecisionDirection.DOWN
                    last_scaling_decision_time = datetime.now(timezone.utc)
                    scaling_decision = ScalingDecision(
                        time=last_scaling_decision_time,
                        direction=scaling_direction,
                        type=scaling_type
                    )
                    logger.info("Made a scaling decision from {} to {} instances: {}".format(
                        current_num_inst, new_num_inst, scaling_decision))
                    self.scaling_decisions.append(scaling_decision)

            except Exception as e:
                logger.exception('Got exception {} on main handler'.format(e))

            # in case the model was trained or evaluation took a while, we still want re-evaluation every X seconds
            # so take out the evaluation time from the overall time
            eval_duration = (datetime.now() - start_time).seconds * 1e+6 + (datetime.now() - start_time).microseconds
            sleep_time = max(0, (self.eval_time_interval_sec * 1e+6 - eval_duration) / (1e+6))
            time.sleep(sleep_time)



if __name__ == '__main__':
    # this loads kubernetes configs (can be set in Configuration class directly or using helper utility)
    config.load_kube_config()

    config_file = os.environ["AUTOSCALER_CONFIG_FILE"]
    controller = Controller(config_file)

    atexit.register(exit_handler)

    controller.start()
