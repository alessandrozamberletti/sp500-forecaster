import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import sqrt
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web


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
            self.__plot_setup()

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
                self.__plot(symbol, current, future, trend, x[-1])

        return x, y

    @staticmethod
    def __normalize(price, time_window):
        return time_window/price - 1

    def plot(self, symbol, current, future, trend, visual_sample):
        self.f.suptitle('Chart&Visual Train Samples - SYMBOL:{0}'.format(symbol))

        self.chart_ax.cla()

        # CURRENT
        self.chart_ax.plot(current[:, -1])
        self.chart_ax.plot([np.average(current)] * self.timestep, color='black', label='current avg price')

        # FUTURE
        xi = range(self.timestep - 1, self.timestep - 1 + self.futurestep)
        color = 'green' if trend else 'red'
        self.chart_ax.plot(xi, future[:, -1], linestyle='--')
        self.chart_ax.plot(xi, [np.average(future)] * len(xi), color=color, label='future avg price')

        # PRESENT|FUTURE SEP
        self.chart_ax.axvline(x=self.timestep - 1, color='gray', linestyle=':')

        self.chart_ax.set_title('Chart')
        self.chart_ax.set_xlabel('days')
        self.chart_ax.set_ylabel('normalized {} price'.format(self.features[-1]))
        self.chart_ax.legend(loc='upper left')

        # VISUAL SAMPLE CHNS
        for i in range(len(self.visual_ax)):
            ax = self.visual_ax[i]
            ax.cla()
            ax.axis('off')
            ax.set_title(self.features[i])
            ax.imshow(visual_sample[:, :, i], cmap='gray')

        plt.show()
        plt.pause(.00001)

    def __plot_setup(self):
        plt.ion()
        self.f = plt.figure()
        gs = gridspec.GridSpec(self.chns, 2)
        self.chart_ax = plt.subplot(gs[:, 0])
        self.visual_ax = []
        for i in range(self.chns):
            self.visual_ax.append(plt.subplot(gs[i, -1]))
