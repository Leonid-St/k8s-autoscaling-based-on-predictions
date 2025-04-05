import xgboost as xgb
import pandas as pd


class XGBoostModel:
    def __init__(self, cluster_metrics):
        self.cpu_model = xgb.XGBRegressor(
            n_estimators=500,  # Увеличиваем количество деревьев
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:quantileerror',
            quantile_alpha=0.9,  # 90-й перцентиль
            n_jobs=-1  # Используем все ядра CPU
        )
        self.memory_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:quantileerror',
            quantile_alpha=0.9,
            n_jobs=-1  # Используем все ядра CPU
        )
        self.cluster_metrics = cluster_metrics
        self._cpu_fitted = False
        self._memory_fitted = False

    def partial_fit(self, df):
        if df.empty:
            raise ValueError("Empty training data")

        # Используем .loc для избежания предупреждений
        cluster_state = self.cluster_metrics.get_cluster_state()
        df = df.copy()
        df.loc[:, 'active_nodes'] = cluster_state['total_nodes']
        df.loc[:, 'avg_cluster_cpu'] = cluster_state['average_cpu']
        df.loc[:, 'avg_cluster_mem'] = cluster_state['average_memory']

        features = self._create_features(df)

        if 'cpu' in df.columns:
            cpu_target = df[['cpu']].values

            if self._cpu_fitted:
                self.cpu_model.fit(
                    features,
                    cpu_target,
                    xgb_model=self.cpu_model.get_booster(),
                    eval_set=[(features, cpu_target)],
                    early_stopping_rounds=10
                )
            else:
                self.cpu_model.fit(features, cpu_target)
                self._cpu_fitted = True

        if 'memory' in df.columns:
            memory_target = df[['memory']].values

            if self._memory_fitted:
                self.memory_model.fit(
                    features,
                    memory_target,
                    xgb_model=self.memory_model.get_booster(),
                    eval_set=[(features, memory_target)],
                    early_stopping_rounds=10
                )
            else:
                self.memory_model.fit(features, memory_target)
                self._memory_fitted = True

    def predict(self, timestamp):
        features = self._create_features(pd.DataFrame({'timestamp': [timestamp]}))
        cpu_pred = self.cpu_model.predict(features)[0]

        # Убираем inverse_transform и ограничения
        return {'cpu': max(0, cpu_pred)}

    def _create_features(self, df):
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'] >= 5
        df['minutes_since_midnight'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        return df[['hour', 'day_of_week', 'month', 'is_weekend', 'minutes_since_midnight']]
