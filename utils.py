from datapackage import Package
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import matplotlib.pyplot as plt
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

        # no data before timestep and after -futurestep
        chart = data['ohlcv']['close'].values[timestep:-futurestep]
        plt.plot(chart, color='black')

        # gt vs predicted
        y_expected = data['y']
        for idx, (pred, gt) in enumerate(zip(y_actual, y_expected)):
            if pred != gt:
                plt.axvspan(idx, idx+1, fc='gray')
                continue
            plt.axvspan(idx, idx+1, fc='green', alpha=.5) if pred else plt.axvspan(idx, idx+1, fc='red', alpha=.5)

        plt.title('Predictions vs Ground Truth - SYMBOL:{}'.format(symbol))
        plt.ylabel('close price')
        plt.xlabel('days')

        legend = [mpatches.Patch(label='positive outlook (gt == pred)', color='green', alpha=.5),
                  mpatches.Patch(label='negative outlook (gt == pred)', color='red', alpha=.5),
                  mpatches.Patch(label='wrong outlook (gt != pred)', color='gray', alpha=.5)]

        plt.legend(handles=legend)

        plt.savefig('out/{}.png'.format(symbol), dpi=300, bbox_inches='tight')
