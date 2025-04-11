import xgboost as xgb
import pandas as pd


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
            self.model.fit(
                features,
                targets,
                xgb_model=self.model.get_booster(),
                eval_set=[(features, targets)],
                early_stopping_rounds=10
            )
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
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'] >= 5
        df['minutes_since_midnight'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        return df[['hour', 'day_of_week', 'month', 'is_weekend', 'minutes_since_midnight']]
