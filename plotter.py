import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class Plotter:
    def __init__(self, features):
        plt.ion()
        self.f = plt.figure()
        self.features = np.array(features)
        gs = gridspec.GridSpec(self.features.shape[0], 2)
        self.chart_ax = plt.subplot(gs[:, 0])
        self.features_ax = [plt.subplot(gs[i, -1]) for i in range(self.features.shape[0])]

    def plot_time_window(self, symbol, current, future, trend, visual_sample):
        self.f.suptitle('Chart&Visual Train Samples - SYMBOL:{0}'.format(symbol))

        self.chart_ax.cla()

        # CURRENT
        self.chart_ax.plot_time_window(current[:, -1])
        self.chart_ax.plot_time_window([np.average(current)] * current.shape[0], color='black', label='current avg price')

        # FUTURE
        xi = np.array(range(current.shape[0] - 1, current.shape[0] - 1 + future.shape[0]))
        color = 'green' if trend else 'red'
        self.chart_ax.plot_time_window(xi, future[:, -1], linestyle='--')
        self.chart_ax.plot_time_window(xi, [np.average(future)] * xi.shape[0], color=color, label='future avg price')

        # PRESENT|FUTURE SEP
        self.chart_ax.axvline(x=current.shape[0] - 1, color='gray', linestyle=':')

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
        plt.pause(.00001)

    def plot_predictions(self, data_test, symbols, timestep, y_actual, y_expected):
        plt.clf()

        for symbol in symbols:
            plt.cla()

            data = data_test[symbol][:, -1]

            cols = []
            for idx, (pt, actual, expected) in enumerate(zip(data, y_actual, y_expected)):
                if actual != expected:
                    cols.append('blue')
                    continue
                cols.append('green') if actual else cols.append('red')

            plt.plot(data, color='black')
            plt.scatter(timestep + np.array(range(len(data[timestep:]))), np.array(data[timestep:]), c=np.array(cols),
                        alpha=0.3)

            from matplotlib.lines import Line2D

            plt.title('Predictions for SYMBOL:{}'.format(symbol))
            plt.ylabel('price')
            plt.xlabel('days')

            legend_els = [Line2D([0], [0], marker='o', color='green', label='positive outlook'),
                          Line2D([0], [0], marker='o', color='red', label='negative outlook'),
                          Line2D([0], [0], marker='o', color='blue', label='wrong prediction')]

            plt.legend(handles=legend_els)

            plt.show()
            plt.pause(10)
