import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
from plotter import Plotter


class DataManager:
    def __init__(self, timestep, futurestep, features, debug=False):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.timestep = timestep
        self.futurestep = futurestep
        self.features = features
        self.chns = len(self.features)
        self.supported_symbols = set([i.encode("utf-8") for i in web.get_iex_symbols()['symbol'].values])

        self.debug = debug
        if self.debug:
            self.plotter = Plotter(features)

    def build_windows(self, symbols):
        symbols = list(set(symbols) & self.supported_symbols)
        assert len(symbols) > 0, 'none of the provided symbols are supported by iex'

        x = []
        y = []
        start = datetime.now() - timedelta(days=2000)
        for symbol in tqdm(symbols, total=len(symbols)):
            # NOTE: extracting too many symbols at once causes error in DataReader parsing
            data = web.DataReader(symbol, data_source='iex', start=start)
            data = data[self.features].values
            data = data[~np.isnan(data).any(axis=1)]

            window_x, window_y = self.__build_time_windows(symbol, data)
            assert len(window_x) == len(window_y), 'non matching samples and targets lengths for {}'.format(symbol)

            x += window_x
            y += window_y

        return np.array(x), np.array(y)

    def __build_time_windows(self, symbol, data):
        x = []
        y = []
        for t in range(0, data.shape[0] - self.timestep - self.futurestep):
            current = data[t:t + self.timestep, :]
            future = data[t + self.timestep - 1:t + self.timestep - 1 + self.futurestep, :]

            current_avg_price = np.average(current)
            future_avg_price = np.average(future)
            trend = future_avg_price > current_avg_price

            p0 = current[0, :]
            current = self.__normalize(p0, current)
            future = self.__normalize(p0, future)

            current = self.scaler.fit_transform(current)

            ssize = int(sqrt(self.timestep))
            x.append(current.reshape(ssize, ssize, self.chns))
            y.append(trend)

            if self.debug:
                future = self.scaler.transform(future)
                self.plotter.plot(symbol, current, future, trend, x[-1])

        return x, y

    @staticmethod
    def __normalize(price, time_window):
        return time_window/price - 1
