import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor

plt.rcParams["figure.figsize"] = (7.5, 4.5)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.90, wspace=0.35, hspace=0.55)

def build_model(data: pd.Series, frequency:str, prediction_length:int) -> Predictor:
    training_data = ListDataset(
        [{"start": data.index[0], "target": data.values}],
        freq=frequency,
        one_dim_target=True
    )
    estimator = DeepAREstimator(freq=frequency, prediction_length=prediction_length, trainer=Trainer(epochs=10))
    predictor = estimator.train(training_data=training_data)
    return predictor

def generate_forecast(predictor, data: pd.Series, frequency:str) -> Forecast:
    test_data = ListDataset(
        [{"start": data[:-1].index[0], "target": data.values}],
        freq=frequency,
        one_dim_target=True
    )
    return next(iter(predictor.predict(test_data)))


def plot_forecast_df(data: pd.Series, forecast_df: pd.Series, filename:str):
    plt.clf()
    plt.ioff()
    data.last('3H').plot(color='b', linewidth=2)
    forecast_df.plot(color='g')
    plt.legend(["observations", "mean prediction"], # "median prediction", ],
               loc="upper center", bbox_to_anchor = (0.5, -0.10), ncol=2)
    plt.savefig(filename)


def plot_forecast(data: pd.Series, forecast, filename:str):
    plt.clf()
    plt.ioff()
    data.last('3H').plot(color='b', linewidth=2, label='observations')
    if forecast is not None:
        forecast.plot(color='g', show_mean=True, prediction_intervals=[50, 95], label="prediction interval ") # prediction_intervals=[50.0, 95.0]) #, output_file=filename)
    plt.legend(["observations",  "median prediction interval", "mean prediction", "95% prediction interval"],
               loc="upper center", fancybox = True, bbox_to_anchor = (0.5, -0.05), ncol=4,
               fontsize=8)
    plt.savefig(filename, dpi=200)

# def train_and_forecast_future_values(data: pd.Series, prediction_time: datetime, frequency:int):
#     # same is done as in the training data
#     test_data = ListDataset(
#         [{"start": df.index[skip_first_entries], "target": df.value[skip_first_entries:]}],
#         freq="1min",
#         one_dim_target=True
#     )
#     pass
