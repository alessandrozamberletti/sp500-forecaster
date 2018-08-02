# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from data_manager import DataManager
import random
import utils

timestep = 144
future_window = 30
debug = True
features = ['open', 'high', 'low']

# RETRIEVE SYMBOLS
print('0) Retrieving SP500 symbols..')

sp500_symbols = utils.sp500_symbols()
sp500_symbols = random.sample(sp500_symbols, 3)

split_pt = int(len(sp500_symbols) * .8)
train_symbols = sp500_symbols[:split_pt]
test_symbols = sp500_symbols[split_pt:]

assert len(train_symbols) > 0 and len(test_symbols), 'no valid symbols found'

print('Found: {} train symbols - {} test symbols'.format(len(train_symbols), len(test_symbols)))

# BUILD TIME WINDOWS
print('1) Splitting into time windows..')

data_manager = DataManager(timestep, future_window, features, debug=debug)
X_train, y_train = data_manager.build_windows(train_symbols)
X_test, y_test = data_manager.build_windows(test_symbols)

assert len(X_train) > 0 and len(X_test) > 0, 'insufficient number of samples'

print('Built: {} train time windows - {} test time windows'.format(len(X_train), len(X_test)))

# BALANCE DATASET
print('2) Balancing data..')

X_train, y_train = utils.balance(X_train, y_train)
X_test, y_test = utils.balance(X_test, y_test)

assert len(X_train) > 0 and len(X_test) > 0, 'insufficient number of samples'

print('Train: {} ↓time windows - {} ↑time windows'.format(len(np.where(y_train == 0)[0]), len(np.where(y_train)[0])))

print('Test: {} ↓time windows - {} ↑time windows'.format(len(np.where(y_test == 0)[0]), len(np.where(y_test)[0])))

# TRAIN MODEL
print('3) Training model..')

ssize = int(sqrt(timestep))
input_size = (ssize, ssize, len(features))
print('Timestep: {} - Futurestep: {} - Input size: {}'.format(timestep, future_window, input_size))

model = utils.cnn(input_size)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, shuffle=True, epochs=10, validation_split=0.2)

# EVAL MODEL
print('Evaluating model..')

# preds = model.predict_classes(X_test)
# # TODO: use zip
# for i in range(len(preds)):
#     pred = preds[i]
#     out = 'OK' if y_test[i] == pred else 'KO'
#     print('expected: {} vs. actual: {} -> {}'.format(y_test[i], bool(pred), out))

print('Test accuracy: {}'.format(model.evaluate(X_test, y_test)[1]))
