import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from plotter import Plotter


class StockDataTransformer:
    """Transform OHLCV data into time windows."""

    def __init__(self, features=['high', 'low', 'close'], timestep=144, futurestep=30, debug=False):
        """
        A StockDataTransformer object is built.
        The object generates time windows by selecting timestep days of features
        columns from OHLCV data.

        Args:
            features (list, optional): OHLCV columns to be included into the time windows, list values must be in
                                       ['open', 'high', 'low', 'close', 'volume'].
            timestep (int, optional): # of past days to be considered when building time windows, default 144.
            futurestep (int, optional): # of future days to be considered when computing forecast trends, default 30.
            debug (bool, optional): print additional debug information during window creation, default False.

        Returns:
            obj: A StockDataTransformer object.
        """
        self.timestep = timestep
        self.futurestep = futurestep
        self.features = features
        hw = int(sqrt(self.timestep))
        self.time_window_shape = (hw, hw, len(self.features))
        if debug:
            self.plotter = Plotter(self.features, self.futurestep)

    def build_train_windows(self, ticker, ohlcv, balance=False):
        """Transform OHLCV data into a training dataset of time windows and future trends.

        Each time window is returned together with a bool value specifying if during the following
        futurestep days the average features[-1] price of the stock increased/decreased
        compared to the past futurestep days.

        Args:
            ticker (str): Ticker name.
            ohlcv (dataframe): Pandas dataframe of OHLCV data satisfying the following condition:
                                all(feat in ohlcv.columns.values for feat in features).
            balance (bool): return an equal number of time windows having positive/negative outlook.

        Returns:
            numpy arrays, list: Time windows are returned as numpy arrays.
                                Future trends are returned as a list of bool values
        """
        ohlcv = self.__select_features(ohlcv)
        x, y = self.__build_time_windows(ticker, ohlcv, self.futurestep)
        self.__validate_shape(x)
        if balance:
            x, y = self.__balance(x, y)
        return x, y

    def build_latest_win(self, ticker, ohlcv):
        """Build the most recent time window (timestep days) for the given OHLCV data.

        Args:
            ticker (str): Ticker name.
            ohlcv (dataframe): Pandas dataframe of OHLCV data satisfying the following condition:
                               all(feat in ohlcv.columns.values for feat in features).

        Returns:
            numpy arrays, list: Time windows are returned as numpy arrays.
        """
        ohlcv = self.__select_features(ohlcv)
        x, _ = self.__build_time_windows(ticker, ohlcv, 0)
        self.__validate_shape(x)
        return np.expand_dims(x[-1, :], axis=0)

    def __build_time_windows(self, symbol, data, futurestep):
        x = []
        y = []
        scaler = MinMaxScaler(feature_range=(0, 1))
        for t in range(0, data.shape[0] - self.timestep - futurestep):
            current = data[t:t + self.timestep, :]
            future = data[t + self.timestep - 1:t + self.timestep - 1 + futurestep, :]

            # if future price > current price -> y=1 else y=0
            current_avg_price = np.average(current[-futurestep:, -1])
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

            if hasattr(self, 'plotter') and futurestep != 0:
                # NOTE: remember not to fit the scaler on future data
                norm_future = self.__scale(p0, future)
                norm_future = scaler.transform(norm_future)
                self.plotter.plot_time_window(symbol, norm_current, norm_future, trend, x[-1])

        return np.array(x), np.array(y)

    def __select_features(self, ohlcv):
        ohlcv = ohlcv[self.features].values
        return ohlcv[~np.isnan(ohlcv).any(axis=1)]

    def __validate_shape(self, windows):
        if windows.shape[0] == 0:
            raise ValueError('insufficient data points')
        elif windows.shape[1:] != self.time_window_shape:
            raise ValueError('data shape: expected {}, found {}'.format(self.time_window_shape, x.shape))

    @staticmethod
    def __balance(x, y):
        false_y_count = y.shape[0] - np.count_nonzero(y)
        true_y_idx, = np.where(y)

        np.random.shuffle(true_y_idx)
        x = np.delete(x, true_y_idx[false_y_count:], axis=0)
        y = np.delete(y, true_y_idx[false_y_count:])

        return x, y

    @staticmethod
    def __scale(p0, tw):
        return tw / p0 - 1
