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
    def __init__(self, features, debug=False):
        self.supported_symbols = set([i.encode("utf-8") for i in web.get_iex_symbols()['symbol'].values])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = features
        # plot training patches creation pipeline
        self.debug = debug
        if self.debug:
            self.plotter = Plotter(features)

    def build_windows(self, symbols, timestep, futurestep):
        symbols = list(set(symbols) & self.supported_symbols)
        assert len(symbols) > 0, 'none of the given symbols are supported by IEX'

        symbols_data = {}
        x = []
        y = []
        # NOTE: IEX data spans back to a maximum of 5 years
        start = datetime.now() - timedelta(days=2000)
        for symbol in tqdm(symbols, total=len(symbols)):
            # NOTE: for IEX extracting too many symbols at once causes error in DailyReader parsing
            ohlcv = web.DataReader(symbol, data_source='iex', start=start)

            data = ohlcv[self.features].values
            data = data[~np.isnan(data).any(axis=1)]

            current, future, current_visual, y_exp = self.__build_time_windows(symbol, data, timestep, futurestep)
            assert len(current_visual) == len(y_exp), 'non matching samples and targets lengths for {}'.format(symbol)

            symbols_data[symbol] = {'ohlcv': ohlcv,
                                    'current': np.array(current),
                                    'future': np.array(future)}
            x += current_visual
            y += y_exp

        return symbols_data, np.array(x), np.array(y)

    def __build_time_windows(self, symbol, data, timestep, futurestep):
        x = []
        y = []
        current_wins = []
        future_wins = []
        for t in range(0, data.shape[0] - timestep - futurestep):
            current = data[t:t + timestep, :]
            future = data[t + timestep - 1:t + timestep - 1 + futurestep, :]

            # if future price > current price -> y=1 else y=0
            current_avg_price = np.average(current[-futurestep:, -1])
            future_avg_price = np.average(future[:, -1])
            trend = future_avg_price > current_avg_price

            # price at position n = (price at position n) / (price at position 0)
            p0 = current[0, :]
            norm_current = self.__scale(p0, current)

            # normalize between 0 and 1
            norm_current = self.scaler.fit_transform(norm_current)

            # transform stock data to image
            hw = int(sqrt(timestep))
            sample_shape = (hw, hw, len(self.features))

            # append to output data
            x.append(norm_current.reshape(sample_shape))
            y.append(trend)
            current_wins.append(current)
            future_wins.append(future)

            if self.debug:
                # NOTE: remember not to fit the scaler on future data
                norm_future = self.__scale(p0, future)
                norm_future = self.scaler.transform(norm_future)
                self.plotter.plot_time_window(symbol, norm_current, norm_future, trend, x[-1])

        return current_wins, future_wins, x, y

    @staticmethod
    def __scale(price, time_window):
        return time_window/price - 1
