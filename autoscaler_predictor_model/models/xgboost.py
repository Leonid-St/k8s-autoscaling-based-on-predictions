import xgboost as xgb
import pandas as pd
import sys
import datetime


# Кастомный callback для записи времени перед каждым блоком из 500 строк
class TimeLoggingCallback(xgb.callback.TrainingCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.iteration_count = 0

    def after_iteration(self, model, epoch, evals_log):
        self.iteration_count += 1
        # Если прошло 500 итераций, записываем время
        if self.iteration_count % 500 == 0:
            with open(self.log_file, 'a') as f:
                f.write(f"\nВремя начала нового блока: {datetime.datetime.now()}\n")
        return False  # Возвращаем False, чтобы обучение продолжалось


class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            n_jobs=-1
        )
        self._fitted = False

    def fit(self, df):
        if df.empty:
            raise ValueError("Empty training data")

        # Создаем признаки
        features = self._create_features(df)

        # Целевые переменные
        targets = df[['cpu', 'memory']].values

        if self._fitted:
            # Открываем файл для записи лога
            log_file = 'training_log_xgboost.txt'
            # Открываем файл для записи лога
            with open(log_file, 'w') as f:
                f.write(f"Начало обучения: {datetime.datetime.now()}\n")
            self.model.fit(
                features,
                targets,
                xgb_model=self.model.get_booster(),
                eval_set=[(features, targets)],
                verbose=True,
               #callbacks=[TimeLoggingCallback(log_file)]
            )
            # Закрываем файл лога
            with open(log_file, 'a') as f:
                f.write(f"Обучение завершено: {datetime.datetime.now()}\n")
        else:
            self.model.fit(features, targets)
            self._fitted = True

    def predict(self, timestamp):
        features = self._create_features(pd.DataFrame({'timestamp': [timestamp]}))
        prediction = self.model.predict(features)
        return {
            'cpu': prediction[0][0],
            'memory': prediction[0][1]
        }

    def _create_features(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['minute'] = df['timestamp'].dt.minute
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        return df[['minute', 'hour', 'day_of_week', 'day_of_month', 'month']]
