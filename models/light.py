import lightgbm as lgb
import pandas as pd
import pickle
from utils import make_train_test, load_data
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 未筛选特征所用到的列
train_cols = ['county', 'target', 'is_business', 'product_type', 'is_consumption', 'datetime', 'row_id']
test_cols = ['county', 'is_business', 'product_type', 'is_consumption', 'prediction_datetime', 'row_id']
train_history_cols = ['target', 'county', 'is_business', 'product_type', 'is_consumption', 'datetime']
client_cols = ['product_type', 'county', 'eic_count', 'installed_capacity', 'is_business', 'date']
gas_cols = ['forecast_date', 'lowest_price_per_mwh', 'highest_price_per_mwh']
electricity_cols = ['forecast_date', 'euros_per_mwh']
forecast_cols = ['latitude', 'longitude', 'hours_ahead', 'temperature', 'dewpoint', 'cloudcover_high', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_total', '10_metre_u_wind_component', '10_metre_v_wind_component', 'forecast_datetime', 'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall', 'total_precipitation']
historical_cols = ['datetime', 'temperature', 'dewpoint', 'rain', 'snowfall', 'surface_pressure','cloudcover_total','cloudcover_low','cloudcover_mid','cloudcover_high','windspeed_10m','winddirection_10m','shortwave_radiation','direct_solar_radiation','diffuse_radiation','latitude','longitude']
station_cols = ['longitude', 'latitude', 'county']
# 读取文件
train = pd.read_csv('../data/train.csv', parse_dates=['datetime'], usecols=train_cols)
train_history = pd.read_csv('../data/train.csv', parse_dates=['datetime'], usecols=train_history_cols)
client = pd.read_csv('../data/client.csv', parse_dates=['date'], usecols=client_cols)
gas_prices = pd.read_csv('../data/gas_prices.csv', parse_dates=['forecast_date'], usecols=gas_cols)
electricity = pd.read_csv('../data/electricity_prices.csv', parse_dates=['forecast_date'], usecols=electricity_cols)
historical_weather = pd.read_csv('../data/historical_weather.csv', parse_dates=['datetime'],
                                 usecols=historical_cols)
forecast_weather = pd.read_csv('../data/forecast_weather.csv', parse_dates=['forecast_datetime'],
                               usecols=forecast_cols)
station = pd.read_csv('../data/weather_station_to_county_mapping.csv', usecols=station_cols)
features, X, Y = load_data(train, train_history, client, gas_prices, electricity, historical_weather, forecast_weather,
                           station)

(X_train, y_train), (X_test, y_test) = make_train_test(X, Y, 1, 0.7)


class LightXGB:
    def __init__(self):
        self.model = lgb.LGBMRegressor(objective='regression', n_estimators=100)

    def fit(self, X_train, y_train, features):
        self.model.fit(X_train, y_train, feature_name=features)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        print(mean_absolute_error(y_pred, y_test))
        return y_pred

    def importance(self, max_num):
        lgb.plot_importance(self.model, max_num_features=max_num)
        plt.show()


model = LightXGB()
model.fit(X_train, y_train, features)
model.predict(X_test)
model.importance(20)