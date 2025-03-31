# from matplotlib import pyplot as plt

# bc = pd.read_csv('resource-25-03-13-12-09.csv', index_col=0, parse_dates=True)
# bc = bc.drop(bc.columns[[1]], axis=1)
#
# print(bc.describe())
# print(bc.info())
# bc.plot(subplots=True, figsize=(12, 8))
#
#
# def plotcharts(y, title, lags=None, figsize=(12, 8)):
#     fig = plt.figure(figsize=figsize)
#     layout = (2, 2)
#     ts_ax = plt.subplot2grid(layout, (0, 0))
#     hist_ax = plt.subplot2grid(layout, (0, 1))
#     acf_ax = plt.subplot2grid(layout, (1, 0))
#     pacf_ax = plt.subplot2grid(layout, (1, 1))
#
#     y.plot(ax=ts_ax)
#     ts_ax.set_title(title, fontsize=14, fontweight="bold")
#     y.plot(ax=hist_ax, kind="hist", bins=25)
#     hist_ax.set_title("Histogram")
#     smt.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax)
#     smt.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax)
#     [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
#     plt.tight_layout()
#     return ts_ax, acf_ax, pacf_ax
#
#
# series = bc
#
# series.plot(subplots=True, figsize=(12, 8))
#
# num_var = len(series.iloc[1, :])
# for i in range(0, num_var):
#     plotcharts(series.iloc[:, i].dropna(), title=series.columns[i], lags=48)
#
# from statsmodels.tsa.stattools import adfuller
#
#
# def ad_test(dataset):
#     dftest = adfuller(dataset, autolag="AIC")
#     print("1. ADF : ", dftest[0])
#     print("2. P-Value : ", dftest[1])
#     print("3. Num Of Lags : ", dftest[2])
#     print("4. Num of Observations Used For ADF Regression and Critical Values Calculation : ", dftest[3])
#     print("5. Critical Values : ")
#     for key, val in dftest[4].items():
#         print("\t", key, ": ", val)
#     print('\n\n')
#
#
# print("--------------------------------------------")
# ad_test(series["cpu"])
# print("--------------------------------------------")
#
# # Преобразование данных в логарифмическую шкалу
# log = pd.DataFrame(np.log(series))
#
# # Изменение значений log
# # 1
# log_diff = log.diff().dropna()
# # 2
# log_diff = log_diff.diff().dropna()
# log_diff.plot(subplots=True)
#
# # импортируем для нормализации
# from sklearn.preprocessing import MinMaxScaler
#
# norm = MinMaxScaler()
#
# print("--------------------------------------------")
#
# # нормализация
# join_norm = pd.DataFrame(norm.fit_transform(log_diff), columns=log_diff.columns)
# print(join_norm)
# print("--------------------------------------------")
#
# print("--------------------------------------------")
# ad_test(join_norm["cpu"])
# print("--------------------------------------------")
#
# num_var = len(join_norm.iloc[1, :])
# for i in range(0, num_var):
#     plotcharts(join_norm.iloc[:, i].dropna(), title=join_norm.columns[i], lags=48)
#
# # разделение датасета на обучающую и тестовую выборку
# # n_obs = 7
#
# n_obs = int(len(join_norm) * 0.9)  # Вычисляем 90% от длины данных
# train = join_norm[:n_obs]  # Берем первые 90% данных
# test = join_norm[n_obs:]  # Оставшиеся 10% данных
#
# # train, test = log_diff[:-n_obs], log_diff[-n_obs:]
# log_diff.head()
#
#
# # from statsmodels.tsa.api import VAR
# #
# # model = VAR(log_diff["cpu"])
# # results = model.fit(maxlags=22, ic='aic')
# # results.summary()
# #
# #
# # lag_order = results.k_ar
# # predicted = results.forecast(log_diff.values[-lag_order:],n_obs)
# # forecast = pd.DataFrame(predicted, index = log_diff.index[-n_obs:], columns = log_diff.columns)
#
#
# # построение предсказанных значений
# # p1 = results.plot_forecast(1)
# # p1.tight_layout()
#
#
# # интвертирование Differencing Transformation
# def invert_transformation(df, df_forecast, second_diff):
#     for col in df.columns:
#         # Отменить 2-е различие
#         if second_diff:
#             df_forecast[str(col)] = (df[col].iloc[-1] - df[col].iloc[-2]) + df_forecast[str(col)].cumsum()
#         # Отменить 1-е различие
#         df_forecast[str(col)] = df[col].iloc[-1] + df_forecast[str(col)].cumsum()
#
#     return df_forecast
#
#
# # forecast_values = invert_transformation(train, forecast, second_diff=True)
#
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from numpy import asarray as arr
#
# # разделение датасета на обучающую и тестовую выборку
# # X, y = log_diff.iloc[:, :], log_diff.iloc[:, :-2]
# #
# # data_dmatrix = xgb.DMatrix(data=X, label=y)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#
# # Используем только столбец 'cpu' для y
# X = log_diff[['cpu']]  # Признаки
# y = log_diff['cpu']  # Целевая переменная
#
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#
# # Создание DMatrix
# data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
#
# xg_reg = xgb.XGBRegressor(
#     objective='reg:squarederror',
#     colsample_bytree=0.3,
#     learning_rate=0.1,
#     max_depth=1,
#     alpha=10,
#     n_estimators=200,
# )
# xg_reg.fit(X_train, y_train)
#
# preds = xg_reg.predict(X_test)
#
# rmse = np.sqrt(mean_squared_error(y_test, preds))
# print("RMSE: %f" % rmse)
# # ### Перекрестная проверка (k-кратная)
#
# # Поскольку XGBoost не специфичен для данных временных рядов, для построения более надежных моделей обычно
# # выполняется k-кратная перекрестная проверка, при которой все записи в исходном наборе обучающих данных используются
# # как для обучения, так и для проверки. Кроме того, каждая запись используется для проверки только один раз. XGBoost
# # поддерживает k-кратную перекрестную проверку с помощью метода cv(). Все, что вам нужно сделать, это указать
# # параметр unfolds, который представляет собой количество наборов перекрестной проверки, которые вы хотите создать.
#
# # <small>[Source](https://www.datacamp.com/community/tutorials/xgboost-in-python/)</small>
#
#
# params = {"objective": "reg:squarederror",
#           'colsample_bytree': 0.3,
#           'learning_rate': 0.1,
#           'max_depth': 2,
#           'alpha': 10}
#
# cv_results = xgb.cv(
#     dtrain=data_dmatrix,
#     params=params,
#     nfold=10,  # Увеличиваем количество фолдов
#     num_boost_round=100,  # Увеличиваем количество итераций
#     early_stopping_rounds=20,  # Увеличиваем раннюю остановку
#     metrics="rmse",
#     as_pandas=True,
#     seed=123,
# )
#
# cv_results.head()
#
# print("print((cv_results['test-rmse-mean']).tail(1))")
# print((cv_results["test-rmse-mean"]).tail(1))
# print("print((cv_results['test-rmse-mean']).tail(1))")
#
# cv_results.plot(subplots=True, figsize=(10, 10))

# # Фактические и прогнозируемые графики
# fig, axes = plt.subplots(nrows=int(len(log_diff.columns) / 2), ncols=3, dpi=100, figsize=(10, 10))
#
# for i, (col, ax) in enumerate(zip(log_diff.columns, axes.flatten())):
#     forecast_values[col].plot(color='#F4511E', legend=True, ax=ax).autoscale(axis=' x', tight=True)
#     test[col].plot(color='#3949AB', legend=True, ax=ax)
#
#     ax.set_title(col + ' - Actual vs Forecast')
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
#
#     ax.spines["top"].set_alpha(0)
#     ax.tick_params(labelsize=6)


## Пророк Facebook

# Библиотека с открытым исходным кодом доступна [здесь](https://facebook.github.io/prophet /)
#
# Facebook prophet был создан для работы в качестве инструмента для большинства общих прогнозов временных рядов. Он
# прост в использовании и интуитивно понятным образом обучает начинающих машинному обучению.
#
# Facebook также позволяет тем, кто разбирается в предметной области, не быть заблокированными, когда они получают
# ответ и получают некоторую дополнительную информацию, которая может принести некоторую пользу.
#
# Если вы используете другие библиотеки прогнозирования. Эти другие библиотеки проделали большую фундаментальную
# работу и прогнозируют, что в библиотеке есть два автоматизированных метода: один - auto.arima, а другой -
# экспоненциальное сглаживание. Они оба выполняют процесс выбора модели, поэтому они действительно пытаются выполнить
# за вас большую работу и устранить много трудностей при построении модели, но иногда вы могли бы получить плохо
# работающие модели, если бы просто применили их к набору данных. Поскольку результаты не всегда достаточно
# интуитивны, чтобы улучшить прогноз.
#
# Facebook prophet способен визуализировать важные особенности временных рядов, такие как тенденции, выбросы,
# сезонность и т.д. Кроме того, метод прогнозирования достаточно надежен, чтобы обрабатывать любые пропущенные значения.
#
# Поэтому, как правило, при решении задач с временными рядами вы хотели бы смоделировать процесс генерации того,
# как будет создаваться этот временной ряд. Это становится трудным для написания, генерирующая модель для процесса
# временных рядов подобна тому, что в каждом состоянии будет возникать новая проблема, и она будет каким-то образом
# зависеть от прошлого. Вместо этого facebook построил дискриминационную модель, которая представляет собой простую
# разложимую модель временных рядов. Это обобщенная аддитивная модель, поэтому каждый компонент является аддитивным,
# но отдельные компоненты могут быть нелинейными.
#
# $$
# y(t) = \text{piecewise_trend}(t) + \text{seasonality}(t) + \text{holiday-effects}(t) + \text{noise}
# $$
#
# Первый компонент - это кусочный тренд, который может быть либо логистическим трендом, либо линейным трендом,
# и это в основном будет определять, насколько быстро растет или уменьшается временной ряд.
#
# Второй компонент - это сезонность, то есть то, что происходит регулярно, циклически. В комплект входят некоторые
# праздничные эффекты и шумоподавление.
#
# Кусочный тренд разрабатывается с использованием L1-регуляризованных сдвигов тренда. Сезонность рассчитывается с
# использованием рядов Фурье. А праздничные эффекты разрабатываются с использованием фиктивных переменных. Они не
# слишком сложны.
#
# Кусочно-линейный тренд или логистический тренд разрабатывается путем генерации набора возможных точек изменения.
# Это точки, в которых модель, по ее мнению, потенциально может изменить свою траекторию, затем они помещают априор
# Лапласа, который похож на разреженный априор, в котором предполагается, что большую часть времени эти изменения
# равны нулю, но иногда это позволит ей измениться. Таким образом, данные, по сути, расскажут нам, когда временной
# ряд изменил свою траекторию, что является действительно приятной особенностью. Итак, prophet учится на основе
# данных, как локально экстраполировать результаты моделирования на основе прошлых данных.
#
# [Источник](https://www.youtube.com/watch?v=pOYAXv15r3A&feature=youtu.be )


# bc = pd.read_csv("resource-25-03-13-12-09.csv")
# bc_cpu = bc.drop(bc.columns[[1]], axis=1)
#
# bc_cpu.columns = ["ds", "y"]
#
# bc_cpu.head()


# построение графиков для каждой серии
# def fit_model(df):
#     m = Prophet(daily_seasonality=True)
#     m.fit(df)
#     fut = m.make_future_dataframe(periods=365)
#     fore = m.predict(fut)
#     return m, fore, fut
#
#
# def fb_plots(m, fore):
#     return plot_plotly(m, fore)
#
#
# def fb_subplots(m, fore):
#     return m.plot(fore), m.plot_components(fore)
#
#
# model, forecast, future = fit_model(bc_cpu)
#
# future.tail()
#
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#
# fb_plots(model, forecast).show()
#
# q, w = fb_subplots(model, forecast)
#
# q.show()
#
# w.show()
# plt.show()
if __name__ == '__main__':
    import holidays
    import pandas as pd
    from prophet import Prophet
    from scipy import stats
    from scipy.special import inv_boxcox

    data = pd.read_csv("resource-25-03-26-12-49.csv")
    data_cpu = data.drop(data.columns[[2]], axis=1)

    # Преобразование столбца с датой в формат datetime
    data_cpu['timestamp'] = pd.to_datetime(data_cpu['timestamp'])

    # Фильтрация данных за март 2025 года
    march_data = data_cpu[(data_cpu['timestamp'].dt.year == 2025) & (data_cpu['timestamp'].dt.month == 3)]

    # Фильтрация данных за март и апрель 2025 года
    march_april_data = data_cpu[(data_cpu['timestamp'].dt.year == 2025) &
                                (data_cpu['timestamp'].dt.month.isin([3, 4]))]

    # Фильтрация данных за апрель 2025 года
    april_data = data_cpu[(data_cpu['timestamp'].dt.year == 2025) &
                          (data_cpu['timestamp'].dt.month.isin([4]))]

    # Фильтрация данных за апрель 2025 года
    april_may_data = data_cpu[(data_cpu['timestamp'].dt.year == 2025) &
                              (data_cpu['timestamp'].dt.month.isin([4, 5]))]

    # Переименование колонки 'timestamp' в 'ds'
    april_may_data = april_may_data.rename(columns={'timestamp': 'ds'})
    # Переименование колонки 'cpu' в 'y'
    april_may_data = april_may_data.rename(columns={'cpu': 'y'})

    DATA = april_may_data.copy()

    print("BEFORE BOXCOX")
    print("-----------------------------------")
    print(DATA.describe())
    print(DATA.info())
    print("-----------------------------------")
    DATA = DATA.copy()
    DATA['y'], lmbd = stats.boxcox(DATA['y'])

    print("AFTER BOXCOX")
    print("-----------------------------------")
    print(DATA.describe())
    print(DATA.info())
    print("-----------------------------------")
    # Кол-во 15 минутных интервалов которые надо отрезать и предсказать
    # 4 = час
    #  4*24 = 96 = день
    predictions = 96

    # Отрезаем из обучающей выборки последние N точек, чтобы измерить на них качество
    train_df = DATA[:-predictions]
    train_df.head()

    april_may_data_df = april_may_data[:-predictions]


    # Вкидываем праздники, для их учёта моделькой
    holidays_dict = holidays.RU(years=2025)
    df_holidays = pd.DataFrame.from_dict(holidays_dict, orient='index') \
        .reset_index()
    df_holidays = df_holidays.rename({'index': 'ds', 0: 'holiday'}, axis='columns')
    df_holidays['ds'] = pd.to_datetime(df_holidays.ds)
    df_holidays = df_holidays.sort_values(by=['ds'])
    df_holidays = df_holidays.reset_index(drop=True)
    print("df_holidays.head()")
    print(df_holidays.head())
    print("df_holidays.head()")
    # Покрутим разные комбинации гиперпараметров
    import itertools
    import numpy as np
    import pandas as pd
    from prophet.diagnostics import cross_validation
    from prophet.diagnostics import performance_metrics
    from prophet.plot import plot_cross_validation_metric

    # param_grid = {
    #     'changepoint_prior_scale': [0.25, 0.05, 0.1],  ## по умолчанию 0.05, попробуем увеличить и уменьшить в два раза
    #     'seasonality_prior_scale': [5.0, 10.0, 20.0],  ## по умолчанию 10.0, попробуем увеличить и уменьшить в два раза
    #     'holidays_prior_scale': [5.0, 10.0, 20.0],  ## по умолчанию 10.0, попробуем увеличить и уменьшить в два раза
    # }

    # Создаем все комбинации параметров
    # all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    # mapes = []  # Сюда будем складывать метрику MAPE

    # Создаем все комбинации параметров
    # all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    # mapes = []  # Сюда будем складывать метрику MAPE

    # period = столько, сколько мы хотим предсказывать
    # horizon = period * 2
    # initial = horizon * 3
    # при этом, периоды нужно расставить так, чтобы в ваш временной ряд влезло несколько прогнозов
    # подробнее тут - https://ranalytics.github.io/tsa-with-r/ch-intro-to-prophet.html#sec-prophet-optimal-model

    # Крутим кроссвалидацию со всеми комбинациями параметров
    # for params in all_params:
    #     m = Prophet(**params, holidays=df_holidays, daily_seasonality=True, weekly_seasonality=True,
    #                 yearly_seasonality="auto").fit(DATA)  # Fit model with given params
    #     df_cv = cross_validation(m, initial='7 days', period='1 days', horizon='2 days', parallel='processes')
    #     df_p = performance_metrics(df_cv,
    #                                rolling_window=1)  # тут окно для подсчета метрики 1, чтобы метрика считалась по
    #     # всему горизонту
    #     mapes.append(df_p['mape'].values[0])
    #
    # # Смотрим на результаты с разными параметрами
    # tuning_results = pd.DataFrame(all_params)
    # tuning_results['mape'] = mapes
    # print(tuning_results)
    # # Отображаем лучшие параметры
    # best_params = all_params[np.argmin(mapes)]
    # print(best_params)
    #     changepoint_prior_scale  seasonality_prior_scale  holidays_prior_scale      mape
    # 0                      0.25                      5.0                   5.0  0.114762
    # 1                      0.25                      5.0                  10.0  0.114762
    # 2                      0.25                      5.0                  20.0  0.114751
    # 3                      0.25                     10.0                   5.0  0.114748
    # 4                      0.25                     10.0                  10.0  0.114755
    # 5                      0.25                     10.0                  20.0  0.114753
    # 6                      0.25                     20.0                   5.0  0.114791
    # 7                      0.25                     20.0                  10.0  0.114796
    # 8                      0.25                     20.0                  20.0  0.114801
    # 9                      0.05                      5.0                   5.0  0.114646
    # 10                     0.05                      5.0                  10.0  0.114627
    # 11                     0.05                      5.0                  20.0  0.114643
    # 12                     0.05                     10.0                   5.0  0.114637
    # 13                     0.05                     10.0                  10.0  0.114637
    # 14                     0.05                     10.0                  20.0  0.114639
    # 15                     0.05                     20.0                   5.0  0.114620
    # 16                     0.05                     20.0                  10.0  0.114619
    # 17                     0.05                     20.0                  20.0  0.114609
    # 18                     0.10                      5.0                   5.0  0.114672
    # 19                     0.10                      5.0                  10.0  0.114674
    # 20                     0.10                      5.0                  20.0  0.114677
    # 21                     0.10                     10.0                   5.0  0.114682
    # 22                     0.10                     10.0                  10.0  0.114680
    # 23                     0.10                     10.0                  20.0  0.114690
    # 24                     0.10                     20.0                   5.0  0.114674
    # 25                     0.10                     20.0                  10.0  0.114674
    # 26                     0.10                     20.0                  20.0  0.114671
    # {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 20.0, 'holidays_prior_scale': 20.0}
    # Настраиваем prophet – говорим ему учитывать праздники и сезонности
    # Подробнее тут - https://ranalytics.github.io/tsa-with-r/ch-intro-to-prophet.html#sec-prophet-seasonal-components
    m = Prophet(
        holidays=df_holidays,
        daily_seasonality="auto",
        weekly_seasonality="auto",
        yearly_seasonality="auto",
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=10,
        holidays_prior_scale=0.00005,
        # growth='flat',
        growth='logistic',
    )

    # Добавление пользовательской сезонности с периодом 96 (15 минут)
    m.add_seasonality(name='quarterly', period=96, fourier_order=5000)
    # m.add_regressor('r0')
    data = train_df.copy()
    data['cap'] = 100  # Максимальная емкость
    m.fit(data)
    # Предсказываем 1 день
    future = m.make_future_dataframe(periods=predictions)  # prediction = 96 частей по 15 минут
    # добавляем регрессор
    # future = future.merge(df_r0, on='ds')
    future['cap'] = 100  # Установка максимальной емкости для будущих значений
    forecast = m.predict(future)
    # Смотрим на фактические ошибки модели

    cmp_df = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(DATA.set_index('ds'))
    cmp_df['e'] = cmp_df['y'] - cmp_df['yhat']
    cmp_df['p'] = 100 * cmp_df['e'] / cmp_df['y']
    print('MAPE (средняя абсолютная ошибка в процентах) – ', np.mean(abs(cmp_df[-predictions:]['p'])), '%')
    print('MAE (средняя абсолютная ошибка) – ', np.mean(abs(cmp_df[-predictions:]['e'])))
    # преобразуем обратно данные и округлим полученные значения
    forecast['yhat'] = round(inv_boxcox(forecast['yhat'], lmbd))
    forecast['yhat_upper'] = round(inv_boxcox(forecast['yhat_upper'], lmbd))
    forecast['yhat_lower'] = round(inv_boxcox(forecast['yhat_lower'], lmbd))
    forecast['trend'] = round(inv_boxcox(forecast['trend'], lmbd))
    # Рисуем график с границами прогноза
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go

    # init_notebook_mode(connected=True)

    iplot([
        go.Scatter(x=april_may_data['ds'], y=april_may_data['y'], name='fact'),
        go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='prediction'),
        go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),
        go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),
        go.Scatter(x=forecast['ds'], y=forecast['trend'], name='trend')
    ])

    # Выгружаем прогноз в эксельку. Спрогнозированное значение лежит в столбце yhat
    forecast.to_excel('./app_forecast.xlsx', sheet_name='Data', index=False)
