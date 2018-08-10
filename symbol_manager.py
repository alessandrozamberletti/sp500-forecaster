import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from datetime import datetime, timedelta
from tqdm import tqdm
from plotter import Plotter
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web


class SymbolManager:
    def __init__(self, symbols, features, timestep, futurestep, debug=False):
        # plot training patches creation pipeline
        self.debug = debug
        if self.debug:
            self.plotter = Plotter(features)

        supported_symbols = set([sym.encode("utf-8") for sym in web.get_iex_symbols()['symbol'].values])
        self.symbols = list(set(symbols) & supported_symbols)
        assert len(symbols) > 0, 'none of the given symbols are supported by IEX'

        self.timestep = timestep
        self.futurestep = futurestep
        self.features = features
        self.symbols_data = self.__build_windows()
        self.x = self.__vectorize(self.symbols_data, 'x')
        self.y = self.__vectorize(self.symbols_data, 'y')

    def __build_windows(self):
        symbols_data = {}
        # NOTE: data spans back to a maximum of 5 years
        start = datetime.now() - timedelta(days=2000)
        for symbol in tqdm(self.symbols, total=len(self.symbols)):
            # NOTE: extracting too many symbols at once causes error in DailyReader parsing
            ohlcv = web.DataReader(symbol, data_source='iex', start=start)

            data = ohlcv[self.features].values
            data = data[~np.isnan(data).any(axis=1)]
            if ohlcv.shape[0] == 0:
                print('no data for {}, skipping'.format(symbol))
                continue

            current, future, x, y = self.__build_time_windows(symbol, data)
            assert len(x) == len(y), 'non matching samples and targets lengths for {}'.format(symbol)

            symbols_data[symbol] = {'ohlcv': ohlcv,
                                    'current': current,
                                    'future': future,
                                    'x': x,
                                    'y': y}

        return symbols_data

    def __build_time_windows(self, symbol, data):
        x = []
        y = []
        current_wins = []
        future_wins = []
        scaler = MinMaxScaler(feature_range=(0, 1))
        for t in range(0, data.shape[0] - self.timestep - self.futurestep):
            current = data[t:t + self.timestep, :]
            future = data[t + self.timestep - 1:t + self.timestep - 1 + self.futurestep, :]

            # if future price > current price -> y=1 else y=0
            current_avg_price = np.average(current[-self.futurestep:, -1])
            future_avg_price = np.average(future[:, -1])
            trend = future_avg_price > current_avg_price

            # price at position n = (price at position n) / (price at position 0)
            p0 = current[0, :]
            norm_current = self.__scale(p0, current)

            # normalize between 0 and 1
            norm_current = scaler.fit_transform(norm_current)

            # transform stock data to image
            hw = int(sqrt(self.timestep))
            sample_shape = (hw, hw, len(self.features))

            # append to output data
            x.append(norm_current.reshape(sample_shape))
            y.append(trend)
            current_wins.append(current)
            future_wins.append(future)

            if self.debug:
                # NOTE: remember not to fit the scaler on future data
                norm_future = self.__scale(p0, future)
                norm_future = scaler.transform(norm_future)
                self.plotter.plot_time_window(symbol, norm_current, norm_future, trend, x[-1])

        return np.array(current_wins), np.array(future_wins), np.array(x), np.array(y)

    def balance(self):
        false_y_count = len(self.y) - np.count_nonzero(np.array(self.y))
        true_y_idx, = np.where(self.y)

        np.random.shuffle(true_y_idx)
        self.x = np.delete(self.x, true_y_idx[false_y_count:], axis=0)
        self.y = np.delete(self.y, true_y_idx[false_y_count:])

        return self

    @staticmethod
    def __vectorize(data, key):
        return np.concatenate([data[key] for _, data in data.items()])

    @staticmethod
    def __scale(price, time_window):
        return time_window/price - 1
