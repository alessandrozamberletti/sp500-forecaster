# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from data_manager import DataManager
import random
import utils

timestep = 144
futurestep = 30
debug = False
features = ['high', 'low', 'close']

# RETRIEVE SYMBOLS
print('0) Retrieving SP500 symbols..')

sp500_symbols = utils.sp500_symbols()
sp500_symbols = random.sample(sp500_symbols, 10)

split_idx = int(len(sp500_symbols) * .8)
train_symbols = sp500_symbols[:split_idx]
test_symbols = sp500_symbols[split_idx:]

assert len(train_symbols) > 0 and len(test_symbols), 'no valid symbols found'

print('FOUND: {} train symbols - {} test symbols'.format(len(train_symbols), len(test_symbols)))

# BUILD TIME WINDOWS
print('1) Splitting into time windows..')

data_manager = DataManager(timestep, futurestep, features, debug=debug)
X_train, y_train = data_manager.build_windows(train_symbols)
X_test, y_test = data_manager.build_windows(test_symbols)

assert len(X_train) > 0 and len(X_test) > 0, 'insufficient number of samples'

print('BUILT: {} train time windows - {} test time windows'.format(len(X_train), len(X_test)))

# BALANCE DATASET
print('2) Balancing data..')

X_train, y_train = utils.balance(X_train, y_train)
# X_test, y_test = utils.balance(X_test, y_test)

assert len(X_train) > 0 and len(X_test) > 0, 'insufficient number of samples'

print('TRAIN: {} ↓time windows - {} ↑time windows'.format(len(np.where(y_train == 0)[0]), len(np.where(y_train)[0])))
print('TEST: {} ↓time windows - {} ↑time windows'.format(len(np.where(y_test == 0)[0]), len(np.where(y_test)[0])))

# TRAIN MODEL
print('3) Training model..')

ssize = int(sqrt(timestep))
input_size = (ssize, ssize, len(features))

print('Timestep: {} - Futurestep: {} - Input size: {}'.format(timestep, futurestep, input_size))

model = utils.cnn(input_size)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, shuffle=True, epochs=10, validation_split=0.2)
utils.plot_loss(hist)

# EVAL MODEL
print('Evaluating model..')

print('Test accuracy: {}'.format(model.evaluate(X_test, y_test)[1]))

from datetime import datetime, timedelta
start = datetime.now() - timedelta(days=2000)
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
supported_symbols = set([i.encode("utf-8") for i in web.get_iex_symbols()['symbol'].values])
test_symbols = list(set(test_symbols) & supported_symbols)

import matplotlib.pyplot as plt
preds = model.predict_classes(X_test)

plt.ion()
for symbol in test_symbols:
    plt.cla()

    data = web.DataReader(symbol, data_source='iex', start=start)
    data = data['low'].values

    cols = []
    for idx, (pt, actual, expected) in enumerate(zip(data, preds, y_test)):
        if actual != expected:
            cols.append('blue')
            continue
        cols.append('green') if actual else cols.append('red')

    plt.plot(data, color='black')
    plt.scatter(timestep + np.array(range(len(data[timestep:]))), np.array(data[timestep:]), c=np.array(cols), alpha=0.3)

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
