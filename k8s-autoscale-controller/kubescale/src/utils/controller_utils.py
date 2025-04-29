from utils.email_util import send_email
from utils.ml_utils import plot_forecast
from enum import Enum
from datetime import datetime
import pandas as pd


class ScalingDecisionType(Enum):
    PROACTIVE = 1
    REACTIVE = 2


class ScalingDecisionDirection(Enum):
    UP = 1
    DOWN = 2


class ScalingDecision:
    def __init__(self, time: datetime, direction: ScalingDecisionDirection, type: ScalingDecisionType):
        self.time = time
        self.direction = direction
        self.type = type

    def __repr__(self) -> str:
        return "Time: [{}], Direction: [{}], Type: [{}]".format(self.time, self.direction, self.type)


def notify_scaling_decision(email_receiver:str, time: datetime, data: pd.Series, deployment_name: str, latest_forecast,
                            metric_value_explanation: str, scaling_explanation: str, proactive: bool):
    filename = "forecast_" + str(time).replace(" ", "_") + ".png"
    plot_forecast(data, latest_forecast, filename)
    explanation = metric_value_explanation + ". \n" + scaling_explanation + "."
    if email_receiver is not None:
        send_email(email_receiver, deployment_name, explanation, filename, proactive)


def generate_predictive_metric_value_logging_text(prediction_time: datetime, current_time: datetime,
                                                  current_num_inst: int, current_metric_value: float,
                                                  predicted_metric_value: float):
    text = "Average metric value at prediction time {} is {} for current number of {} instances, " \
           "whilst at current time {} the metric value is {} for {} instances".format(prediction_time,
                round(predicted_metric_value,3), current_num_inst, current_time, round(current_metric_value,3),
                current_num_inst)
    return text


def generate_reactive_metric_value_logging_text(current_time: datetime, current_num_inst: int,
                                                current_metric_value: float):
    text = "Average metric value at current time {} is {} for {} instances". \
        format(current_time, round(current_metric_value,3), current_num_inst)
    return text


def generate_scaling_logging_text(type: ScalingDecisionType, new_num_inst: int, current_num_inst: int,
                                  current_time: datetime, forecast_only: bool = False):
    if forecast_only:
        forecast_only_text = "Forecast only as predictive scaling is disable: "
    else:
        forecast_only_text = ""
    if current_num_inst > new_num_inst:
        direction = "down"
    else:
        direction = "up"
    if type == ScalingDecisionType.PROACTIVE:
        type_text = "Proactive"
    elif type == ScalingDecisionType.REACTIVE:
        type_text = "Reactive"
    else:
        raise ValueError()

    text = "{} {} scale {} to {} instances from current {} instances at time {}".format(
        forecast_only_text, type_text, direction, new_num_inst, current_num_inst, current_time)
    return text
