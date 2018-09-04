import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class Plotter:
    def __init__(self, features, futurestep):
        plt.ion()
        self.f = plt.figure()
        self.features = np.array(features)
        self.futurestep = futurestep
        gs = gridspec.GridSpec(self.features.shape[0], 2)
        self.chart_ax = plt.subplot(gs[:, 0])
        self.features_ax = [plt.subplot(gs[i, -1]) for i in range(self.features.shape[0])]

    def plot_time_window(self, symbol, current, future, trend, visual_sample):
        self.f.suptitle('Chart&Visual Train Samples - SYMBOL:{0}'.format(symbol))

        self.chart_ax.cla()

        # CURRENT
        self.chart_ax.plot(current[:, -1])
        xi = range(current.shape[0] - self.futurestep, current.shape[0])
        self.chart_ax.plot(xi,
                           [np.average(current[-self.futurestep:, -1])] * self.futurestep,
                           color='black',
                           label='past {}d avg price'.format(self.futurestep))

        # FUTURE
        xi = np.array(range(current.shape[0] - 1, current.shape[0] - 1 + future.shape[0]))
        color = 'green' if trend else 'red'
        self.chart_ax.plot(xi,
                           future[:, -1],
                           linestyle='--')
        self.chart_ax.plot(xi,
                           [np.average(future[:, -1])] * xi.shape[0],
                           color=color,
                           label='future {}d avg price'.format(self.futurestep))

        # PRESENT|FUTURE SEP
        self.chart_ax.axvline(x=current.shape[0] - 1,
                              color='gray',
                              linestyle=':')

        # VISUAL SAMPLE CHNS
        for idx, (ax, feature) in enumerate(zip(self.features_ax, self.features)):
            ax.cla()
            ax.axis('off')
            ax.set_title(feature)
            ax.imshow(visual_sample[:, :, idx], cmap='gray')

        # LABELS
        self.chart_ax.set_title('Chart')
        self.chart_ax.set_xlabel('days')
        self.chart_ax.set_ylabel('normalized {} price'.format(self.features[-1]))
        self.chart_ax.legend(loc='upper left')

        plt.show()
        plt.pause(.0001)
