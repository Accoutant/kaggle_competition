import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle
from functools import reduce


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

# 筛选后特征所用到的列
train_cols_selected = ['county', 'target', 'is_business', 'product_type', 'is_consumption', 'datetime', 'row_id']
test_cols_selected = ['county', 'is_business', 'product_type', 'is_consumption', 'prediction_datetime', 'row_id']
train_history_cols_selected = ['target', 'county', 'is_business', 'product_type', 'is_consumption', 'datetime']
client_cols_selected = ['product_type', 'county', 'eic_count', 'installed_capacity', 'is_business', 'date']
gas_cols_selected = ['forecast_date', 'lowest_price_per_mwh', 'highest_price_per_mwh']
electricity_cols_selected = ['forecast_date', 'euros_per_mwh']
forecast_cols_selected = ['latitude', 'longitude', 'hours_ahead', 'temperature', 'dewpoint', 'cloudcover_low', 'cloudcover_total', 'forecast_datetime', 'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall']
historical_cols_selected = ['datetime', 'temperature', 'dewpoint', 'snowfall', 'cloudcover_total','cloudcover_low','shortwave_radiation','direct_solar_radiation','diffuse_radiation','latitude','longitude']
station_cols_selected = ['longitude', 'latitude', 'county']


def structure_time(data, date_col, target_col, day):
    """构造时间位移函数"""
    data_new = data.copy()
    data_new[date_col] = data_new[date_col] + pd.Timedelta(day, 'D')
    data_new.rename(columns={target_col: target_col+'_'+str(day)}, inplace=True)
    return data_new


def merge_data(train, train_history, client, gas_prices, electricity, historical_weather, forecast_weather, station):
    """处理train数据"""
    if 'datetime' in train.columns:
        pass
    else:
        train.rename(columns={'prediction_datetime': 'datetime'}, inplace=True)
    train = split_date(train, 'datetime')  # 拆分时间

    """构造target 2、3、4、5、6、7、14天数据"""
    target2 = structure_time(train_history[['datetime', 'county', 'is_business', 'product_type', 'is_consumption', 'target']],
                             'datetime', 'target', 2)
    target3 = structure_time(train_history[['datetime', 'county', 'is_business', 'product_type', 'is_consumption', 'target']],
                             'datetime', 'target', 3)
    target4 = structure_time(train_history[['datetime', 'county', 'is_business', 'product_type', 'is_consumption', 'target']],
                             'datetime', 'target', 4)
    target5 = structure_time(train_history[['datetime', 'county', 'is_business', 'product_type', 'is_consumption', 'target']],
                             'datetime', 'target', 5)
    target6 = structure_time(train_history[['datetime', 'county', 'is_business', 'product_type', 'is_consumption', 'target']],
                             'datetime', 'target', 6)
    target7 = structure_time(train_history[['datetime', 'county', 'is_business', 'product_type', 'is_consumption', 'target']],
                             'datetime', 'target', 7)
    target14 = structure_time(train_history[['datetime', 'county', 'is_business', 'product_type', 'is_consumption', 'target']],
                              'datetime', 'target', 14)

    """构造target过去第2天的均值，按county取平均"""
    target_mean = target2.groupby(by=['datetime', 'is_business', 'product_type', 'is_consumption']).mean()['target_2']
    target_mean.rename('target_mean', inplace=True)

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
    historical_weather['latitude'] = historical_weather['latitude'].astype(float)            # 转为float类型才能取一位小数
    historical_weather['longitude'] = historical_weather['longitude'].astype(float)
    historical_weather.loc[:, 'latitude'] = historical_weather.loc[:, 'latitude'].round(1)   # 将经纬度取一位小数
    historical_weather.loc[:, 'longitude'] = historical_weather.loc[:, 'longitude'].round(1)
    station['latitude'] = station['latitude'].astype(float)
    station['longitude'] = station['longitude'].astype(float)
    station.loc[:, 'longitude'] = station.loc[:, 'longitude'].round(1)
    station.loc[:, 'latitude'] = station.loc[:, 'latitude'].round(1)
    # 与station数据按照经纬度和时间拼接
    historical_weather = pd.merge(left=historical_weather, right=station, how='left', on=['latitude', 'longitude'])
    # historical_weather.dropna(subset='county', inplace=True)
    historical_weather.drop(columns=['latitude', 'longitude'], inplace=True)
    # 由于一个county对应多个天气站点，将同一个county同一时间的数据平均
    historical_weather_local = historical_weather.groupby(by=['datetime', 'county']).mean()

    """构造全局天气均值"""
    historical_weather_date = historical_weather[['datetime', 'temperature', 'dewpoint', 'snowfall', 'cloudcover_total',
                                                  'cloudcover_low', 'shortwave_radiation', 'direct_solar_radiation',
                                                  'diffuse_radiation']].groupby(by=['datetime']).mean()
    col_new = {}
    for col in historical_weather_date.columns:
        col_new[col] = col + '_global'
    historical_weather_date.rename(columns=col_new, inplace=True)

    """处理forecast_weather数据"""
    forecast_weather = forecast_weather[forecast_weather['hours_ahead'] >= 24]
    forecast_weather['latitude'] = forecast_weather['latitude'].astype(float)            # 转为float类型才能取一位小数
    forecast_weather['longitude'] = forecast_weather['longitude'].astype(float)
    forecast_weather.loc[:, 'longitude'] = forecast_weather.loc[:, 'longitude'].round(1)
    forecast_weather.loc[:, 'latitude'] = forecast_weather.loc[:, 'latitude'].round(1)
    forecast_weather = pd.merge(left=forecast_weather, right=station, on=['latitude', 'longitude'])
    # 去除缺失值以及删除无用列
    # forecast_weather.dropna(subset='county', inplace=True)
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
    data = reduce(lambda left, right: pd.merge(left, right, how='left',
                  on=['datetime', 'county', 'is_business', 'product_type', 'is_consumption']),
                  [train, target2, target3, target4, target5, target6, target7, target14])
    data = pd.merge(left=data, right=target_mean, how='left', on=['datetime', 'is_business',
                                                                  'product_type', 'is_consumption'])
    data = pd.merge(left=data, right=client, how='left', on=['product_type', 'county',
                                                             'is_business', 'year', 'month', 'day'])
    data = pd.merge(left=data, right=gas_prices, how='left', on=['year', 'month', 'day'])
    data = pd.merge(left=data, right=electricity, how='left', on='datetime')
    data = pd.merge(left=data, right=historical_weather_local, how='left', on=['datetime', 'county'])
    data = pd.merge(left=data, right=historical_weather_date, how='left', on=['datetime'])
    data = pd.merge(left=data, right=forecast_weather, how='left', on=['datetime', 'county'])

    return data


def load_data(train, train_history, client, gas_prices, electricity, historical_weather, forecast_weather,
                      station, is_train=True):
    """加载数据"""
    data = merge_data(train, train_history, client, gas_prices, electricity, historical_weather, forecast_weather,
                      station)
    # 删除多余列
    data.drop(columns=['datetime'], inplace=True)
    # one-hot编码
    # data = pd.get_dummies(data, columns=['is_business', 'product_type', 'is_consumption'], dtype=float)
    # 生成nparray数组
    if is_train:
        # 删除缺失值
        data.dropna(subset=['target'], inplace=True)
        X = data.drop(columns=['row_id', 'target'])
        features = list(X.columns)
        X = np.array(X)
        print('X.shape', X.shape)
        Y = data['target']
        Y = np.array(Y)
        output = (features, X, Y)
        with open("train_data.pkl", 'wb') as f:
            pickle.dump(output, f)
    else:
        X = data.drop(columns=['row_id'])
        X = np.array(X)
        print('X.shape', X.shape)
        output = X
        with open("test_data.pkl", 'wb') as f:
            pickle.dump(output, f)
    return output


def make_train_test(X, Y, seed, rate):
    """划分训练集和测试集，并且打乱"""
    idx = int(rate * X.shape[0])
    X_train = X[:idx]
    Y_train = Y[:idx]
    X_test = X[idx:]
    Y_test = Y[idx:]
    shuffled_indices = np.arange(X_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffled_indices)
    X_train, Y_train = X_train[shuffled_indices], Y_train[shuffled_indices]
    return (X_train, Y_train), (X_test, Y_test)


