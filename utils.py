from datapackage import Package
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
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


def vectorize(symbols_data, key):
    return np.concatenate([data[key] for _, data in symbols_data.items()])


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


def cnn(x, y, input_shape):
    model = Sequential()
    model.add(Conv2D(8, (2, 2), input_shape=input_shape))
    model.add(Conv2D(12, (2, 2)))
    model.add(Conv2D(16, (2, 2)))
    model.add(Conv2D(20, (2, 2)))
    model.add(Conv2D(24, (2, 2)))
    model.add(Conv2D(28, (2, 2)))
    model.add(Conv2D(32, (2, 2)))
    model.add(Conv2D(36, (2, 2)))
    model.add(Conv2D(40, (2, 2)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(units=1, activation='sigmoid'))

    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    hist = model.fit(x, y, shuffle=True, epochs=1, validation_split=0.2, callbacks=[es])

    return model, hist


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

        y_expected = data['y']
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
