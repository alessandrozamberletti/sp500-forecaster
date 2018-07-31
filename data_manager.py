import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import sqrt


class DataManager:
    def __init__(self, columns, timestep, futurestep, debug=False):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.columns = columns
        self.chns = len(columns)
        self.timestep = timestep
        self.futurestep = futurestep
        self.debug = debug
        if self.debug:
            self.__plot_setup()

    def __plot_setup(self):
        plt.ion()
        self.f = plt.figure()
        gs = gridspec.GridSpec(3, 2)
        self.chart_ax = plt.subplot(gs[:, 0])
        self.visual_ax = []
        for i in range(self.chns):
            self.visual_ax.append(plt.subplot(gs[i, -1]))

    def build_samples(self, stock, ohlc):
        x = []
        y = []
        for t in range(0, ohlc.shape[0] - self.timestep - self.futurestep):
            current = ohlc[t:t + self.timestep, 1:4]
            future = ohlc[t + self.timestep - 1:t + self.timestep - 1 + self.futurestep, 1:4]

            future_avg_price = np.average(future)
            current_avg_price = np.average(current[:, -1])
            trend = future_avg_price > current_avg_price

            p0 = current[0, :]
            current = self.__normalize(p0, current)
            future = self.__normalize(p0, future)

            current = self.scaler.fit_transform(current)
            future = self.scaler.transform(future)

            ssize = int(sqrt(self.timestep))
            x.append(current.reshape(ssize, ssize, self.chns))
            y.append(trend)

            if self.debug:
                self.f.suptitle('Chart&Visual Train Samples - SYMBOL:{0}'.format(stock))

                self.chart_ax.cla()

                self.chart_ax.plot(current[:, -1])
                self.chart_ax.plot([np.average(current) for _ in range(self.timestep)],
                                   color='black',
                                   label='current avg price')

                xi = range(self.timestep - 1, self.timestep - 1 + self.future_window)
                color = 'green' if trend else 'red'
                self.chart_ax.plot(xi, future[:, -1],
                                   linestyle='--')
                self.chart_ax.plot(xi, [np.average(future) for _ in range(len(xi))],
                                   color=color,
                                   label='future avg price')

                # PRESENT|FUTURE LINE
                self.chart_ax.axvline(x=self.timestep - 1,
                                      color='gray',
                                      linestyle=':')

                self.chart_ax.set_title('Chart')
                self.chart_ax.set_xlabel('days')
                self.chart_ax.set_ylabel('normalized closing price')
                self.chart_ax.legend(loc='upper left')

                for i in range(len(self.visual_ax)):
                    ax = self.visual_ax[i]
                    ax.cla()
                    ax.axis('off')
                    ax.set_title(self.visual_ax_titles[i])
                    ax.imshow(x[-1][:, :, i],
                              cmap='gray')

                plt.show()
                plt.pause(.0001)
        return x, y

    def __normalize(self, price, time_window):
        return np.array([[(i[0] / price[j]) - 1 for j in range(self.chns)] for i in time_window])
