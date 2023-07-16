import pandas as pd
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from ta.volume import AccDistIndexIndicator
from ta import add_all_ta_features
from ta.utils import dropna
from gtda.time_series import SlidingWindow, Stationarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_dataset(path):

    lag_length, horizon = 32, 2

    data = pd.read_csv('stock_alldays.csv').set_index('time')
    # data = data.drop(columns=['unix', 'symbol', 'Volume BTC'])
    data = data[data.name == 'KTC']
    data = data[['open','high','low','close','volume']]
    prices_cols = data.columns


    # Clean NaN values
    data = dropna(data)

    # Add ta features filling NaN values
    data['ao'] = AwesomeOscillatorIndicator(high=data['high'], low=data['low'], fillna=True).awesome_oscillator()
    data['adi'] = AccDistIndexIndicator(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'],
                                        fillna=True).acc_dist_index()
    data['atr'] = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'],
                                fillna=True).average_true_range()

    adx_indicator = ADXIndicator(high=data['high'], low=data['low'], close=data['close'], fillna=True)
    data['adx'] = adx_indicator.adx()
    data['adx_pos'] = adx_indicator.adx_pos()
    data['adx_neg'] = adx_indicator.adx_neg()

    data = data[27:]
    indicator_cols = data.columns[len(prices_cols):]

        # Split data -- train:val:test == 6:2:2
    train_cutoff, val_cutoff = int(len(data) * 0.6), int(len(data) * 0.8)
    train_data, val_data, test_data = data[:train_cutoff], data[train_cutoff:val_cutoff], data[val_cutoff:]

    stationariser = Stationarizer()
    train_data[1:][prices_cols] = stationariser.fit_transform(train_data[prices_cols])
    val_data[1:][prices_cols] = stationariser.transform(val_data[prices_cols])
    test_data[1:][prices_cols] = stationariser.transform(test_data[prices_cols])

    train_data, val_data, test_data = train_data[1:], val_data[1:], test_data[1:]

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    # test_data = scaler.transform(test_data)

    # Segment into train/val/test examples (may use stride < size for train_data only but may cause data leak)
    windows = SlidingWindow(size=lag_length + horizon, stride=2)
    train_data = windows.fit_transform(train_data)
    windows = SlidingWindow(size=lag_length + horizon, stride=lag_length + horizon)
    val_data = windows.fit_transform(val_data)
    # test_data = windows.transform(test_data)
    test_data = windows.fit_transform(test_data)

    # Split all time series segments into x and y
    X_train, y_train = train_data[:, :-horizon, :], train_data[:, -horizon:, :]
    X_val, y_val = val_data[:, :-horizon, :], val_data[:, -horizon:, :]
    X_test, y_test = test_data[:, :-horizon, :], test_data[:, -horizon:, :]

    return X_train,y_train,X_val,y_val,X_test,y_test
