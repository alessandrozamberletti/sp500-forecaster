import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from plotter import Plotter


class StockDataTransformer:
    def __init__(self, features, timestep, futurestep, debug=False):
        self.timestep = timestep
        self.futurestep = futurestep
        self.features = features
        hw = int(sqrt(self.timestep))
        self.time_window_shape = (hw, hw, len(self.features))
        if debug:
            self.plotter = Plotter(self.features, self.futurestep)

    def split_time_windows(self, ticker, ohlcv, balance=False):
        # select feature columns
        ohlcv = ohlcv[self.features].values
        ohlcv = ohlcv[~np.isnan(ohlcv).any(axis=1)]

        x, y = self.__build_time_windows(ticker, ohlcv)
        if x.shape[0] == 0 or x.shape[1:] != self.time_window_shape:
            raise ValueError('wrong data shape: expected {}, found {}'.format(self.time_window_shape, x.shape))

        return self.__balance(x, y) if balance else x, y

    def __build_time_windows(self, symbol, data):
        x = []
        y = []
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

            # append to output data
            x.append(norm_current.reshape(self.time_window_shape))
            y.append(trend)

            if self.plotter:
                # NOTE: remember not to fit the scaler on future data
                norm_future = self.__scale(p0, future)
                norm_future = scaler.transform(norm_future)
                self.plotter.plot_time_window(symbol, norm_current, norm_future, trend, x[-1])

        return np.array(x), np.array(y)

    @staticmethod
    def __balance(x, y):
        false_y_count = len(y) - np.count_nonzero(np.array(y))
        true_y_idx, = np.where(y)

        np.random.shuffle(true_y_idx)
        x = np.delete(x, true_y_idx[false_y_count:], axis=0)
        y = np.delete(y, true_y_idx[false_y_count:])

        return x, y

    @staticmethod
    def __scale(p0, tw):
        return tw / p0 - 1
