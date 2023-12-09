import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle


def split_date(data: DataFrame, col: str):
    """
    拆分时间
    :param data:需要拆分的数据
    :param col: 时间所在的列
    :return: DataFrame
    """
    data['year'] = data[col].apply(lambda x: x.year)
    data['month'] = data[col].apply(lambda x: x.month)
    data['day'] = data[col].apply(lambda x: x.day)
    data['hour'] = data[col].apply(lambda x: x.hour)
    return data


train_cols = ['county', 'target', 'is_business', 'product_type', 'is_consumption', 'datetime', 'row_id']
test_cols = ['county', 'is_business', 'product_type', 'is_consumption', 'datetime', 'row_id']
train_history_cols = ['target', 'county', 'is_business', 'product_type', 'is_consumption', 'datetime']
client_cols = ['product_type', 'county', 'eic_count', 'installed_capacity', 'is_business', 'date']
gas_cols = ['forecast_date', 'lowest_price_per_mwh', 'highest_price_per_mwh']
electricity_cols = ['forecast_date', 'euros_per_mwh']
forecast_cols = ['latitude', 'longitude', 'hours_ahead', 'temperature', 'dewpoint', 'cloudcover_high', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_total', '10_metre_u_wind_component', '10_metre_v_wind_component', 'forecast_datetime', 'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall', 'total_precipitation']
historical_cols = ['datetime', 'temperature', 'dewpoint', 'rain', 'snowfall', 'surface_pressure','cloudcover_total','cloudcover_low','cloudcover_mid','cloudcover_high','windspeed_10m','winddirection_10m','shortwave_radiation','direct_solar_radiation','diffuse_radiation','latitude','longitude']
station_cols = ['longitude', 'latitude', 'county']


def merge_data(train, train_history, client, gas_prices, electricity, historical_weather, forecast_weather, station):
    """处理train数据"""
    if 'datetime' in train.columns:
        pass
    else:
        train.rename(columns={'prediction_datetime': 'datetime'}, inplace=True)
    train = split_date(train, 'datetime')  # 拆分时间

    """处理train_history数据"""
    train_history['datetime'] = train_history['datetime'].apply(lambda x: x + pd.Timedelta(2, 'D'))
    train_history.rename(columns={'target': 'target_used'}, inplace=True)

    """处理client数据"""
    client['datetime'] = client['date'].apply(lambda x: x+pd.Timedelta(2, 'D'))  # 将day向前移两天
    client = split_date(client, 'datetime')
    client.drop(columns=['hour', 'datetime', 'date'], inplace=True)

    """处理gas_prices数据"""
    gas_prices['datetime'] = gas_prices['forecast_date'].apply(lambda x: x + pd.Timedelta(1, 'D'))
    gas_prices = split_date(gas_prices, 'datetime')
    gas_prices.drop(columns=['datetime', 'forecast_date', 'hour'], inplace=True)

    """处理electricity数据"""
    electricity['datetime'] = electricity['forecast_date'].apply(lambda x: x + pd.Timedelta(1, 'D'))
    electricity.drop(columns=['forecast_date'], inplace=True)

    """处理historical_weather数据"""
    historical_weather['datetime'] = historical_weather['datetime'].apply(lambda x: x + pd.Timedelta(37, 'H'))
    historical_weather['latitude'] = historical_weather['latitude'].round(1)   # 将经纬度取一位小数
    historical_weather['longitude'] = historical_weather['longitude'].round(1)
    station.loc[:, 'longitude'] = station.loc[:, 'longitude'].round(1)
    station.loc[:, 'latitude'] = station.loc[:, 'latitude'].round(1)
    # 与station数据按照经纬度和时间拼接
    historical_weather = pd.merge(left=historical_weather, right=station, how='left', on=['latitude', 'longitude'])
    historical_weather.dropna(subset='county', inplace=True)
    historical_weather.drop(columns=['latitude', 'longitude'], inplace=True)
    # 由于一个county对应多个天气站点，将同一个county同一时间的数据平均
    historical_weather = historical_weather.groupby(by=['datetime', 'county']).mean()

    """处理forecast_weather数据"""
    forecast_weather = forecast_weather[forecast_weather['hours_ahead'] >= 24]
    forecast_weather.loc[:, 'longitude'] = forecast_weather.loc[:, 'longitude'].round(1)
    forecast_weather.loc[:, 'latitude'] = forecast_weather.loc[:, 'latitude'].round(1)
    forecast_weather = pd.merge(left=forecast_weather, right=station, on=['latitude', 'longitude'])
    # 去除缺失值以及删除无用列
    forecast_weather.dropna(subset='county', inplace=True)
    forecast_weather.drop(
        columns=['latitude', 'longitude', 'hours_ahead'],
        inplace=True)
    # 将forecast列索引重命名，以防止合并后与historical重名
    forecast_cols_new = {}
    forecast_cols = forecast_weather.columns
    for index in forecast_cols:
        if index == 'forecast_datetime':
            index_new = 'datetime'
        elif index == 'county':
            index_new = index
        else:
            index_new = str(index) + '_fw'
        forecast_cols_new[index] = index_new
    forecast_weather.rename(columns=forecast_cols_new, inplace=True)
    # 去除时间UTC值
    forecast_weather['datetime'] = pd.to_datetime(forecast_weather.datetime).dt.tz_localize(None)
    # 由于一个county对应多个天气站点，将同一个county同一时间的数据平均
    forecast_weather = forecast_weather.groupby(by=['datetime', 'county']).mean()

    """开始拼接数据"""
    data = pd.merge(left=train, right=train_history, how='left', on=['datetime', 'county', 'is_business', 'product_type',
                                                                     'is_consumption'])
    data = pd.merge(left=data, right=client, how='left', on=['product_type', 'county',
                                                             'is_business', 'year', 'month', 'day'])
    data = pd.merge(left=data, right=gas_prices, how='left', on=['year', 'month', 'day'])
    data = pd.merge(left=data, right=electricity, how='left', on='datetime')
    data = pd.merge(left=data, right=historical_weather, how='left', on=['datetime', 'county'])
    data = pd.merge(left=data, right=forecast_weather, how='left', on=['datetime', 'county'])

    return data


def load_data(train, train_history, client, gas_prices, electricity, historical_weather, forecast_weather,
                      station, is_train=True):
    data = merge_data(train, train_history, client, gas_prices, electricity, historical_weather, forecast_weather,
                      station)
    # 删除缺失值
    data.dropna(how='any', inplace=True)
    # 删除多余列
    data.drop(columns=['datetime'], inplace=True)
    # one-hot编码
    data = pd.get_dummies(data, columns=['is_business', 'product_type', 'is_consumption'], dtype=float)
    print(data.columns)
    print(data)
    # 生成nparray数组
    row_id = data['row_id']
    row_id = np.array(row_id)
    if is_train:             # 删除多余列
        X = data.drop(columns=['row_id', 'target'])
        X = np.array(X)
        print('X.shape', X.shape)
        Y = data['target']
        Y = np.array(Y)
        with open("train_data.pkl", 'wb') as f:
            pickle.dump((row_id, X, Y), f)
    else:
        X = data.drop(columns=['row_id'])
        X = np.array(X)
        print('X.shape', X.shape)
        with open("test_data.pkl", 'wb') as f:
            pickle.dump((row_id, X), f)


if __name__ == '__main__':
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
    load_data(train, train_history, client, gas_prices, electricity, historical_weather, forecast_weather, station)