from datapackage import Package
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


def sp500_symbols():
    sp500 = []
    package = Package('https://datahub.io/core/s-and-p-500-companies/datapackage.json')
    for resource in package.resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            sp500 = [s[0].encode('utf-8') for s in resource.read()]

    return sp500


def split(data, ratio):
    split_idx = int(len(data) * ratio)
    return data[:split_idx], data[split_idx:]


def balance(x, y):
    false_y_count = len(y) - np.count_nonzero(np.array(y))
    true_y_idx, = np.where(y)

    np.random.shuffle(true_y_idx)
    x = np.delete(x, true_y_idx[false_y_count:], axis=0)
    y = np.delete(y, true_y_idx[false_y_count:])

    return x, y


def cnn(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(units=1, activation='sigmoid'))

    return model


def plot_loss(data):
    plt.figure()
    plt.ion()

    plt.plot(data.history['loss'], label='train')
    plt.plot(data.history['val_loss'], label='validation')

    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()

    plt.show()
    plt.pause(0.0001)


def save_predictions(symbols_data, timestep, futurestep, y_actual):
    for symbol, data in symbols_data.items():
        plt.cla()

        chart = data['ohlcv']['close'].values[timestep:]
        plt.plot(chart, color='black')

        current_avg = np.average(data['current'][:, -futurestep:, -1], axis=1)
        plt.plot(current_avg, color='pink')

        future_avg = np.average(data['future'][:, :, -1], axis=1)
        plt.plot(future_avg, color='cyan')

        y_expected = data['trend']
        for idx, (y, cur, fut, pred, gt) in enumerate(zip(chart, current_avg, future_avg, y_actual, y_expected)):
            if pred != gt:
                plt.axvspan(idx, idx+1, facecolor='blue', alpha=.5)
                continue
            if pred:
                plt.axvspan(idx, idx+1, facecolor='green', alpha=.5)
            else:
                plt.axvspan(idx, idx+1, facecolor='red', alpha=.5)

        plt.title('Predictions vs Ground Truth - SYMBOL:{}'.format(symbol))
        plt.ylabel('price')
        plt.xlabel('days')

        legend = [mpatches.Patch(color='green', label='gt == pred == positive outlook'),
                  mpatches.Patch(color='red', label='gt == pred == negative outlook'),
                  mpatches.Patch(color='blue', label='gt != pred, wrong outlook'),
                  Line2D([], [], color='pink', label='past {} days avg. price'.format(futurestep)),
                  Line2D([], [], color='cyan', label='future {} days avg. price'.format(futurestep))]

        plt.legend(handles=legend)

        plt.savefig('out/{}.png'.format(symbol), bbox_inches='tight')
