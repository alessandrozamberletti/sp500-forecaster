# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from data_manager import DataManager
from keras.callbacks import EarlyStopping
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

train_symbols, test_symbols = utils.split(sp500_symbols, .8)

assert len(train_symbols) > 0 and len(test_symbols), 'no valid symbols found'

print('SELECTED: {} train symbols - {} test symbols'.format(len(train_symbols), len(test_symbols)))

# BUILD TIME WINDOWS
print('1) Splitting into time windows..')

data_manager = DataManager(timestep, futurestep, features, debug=debug)
_, X_train, y_train = data_manager.build_windows(train_symbols)

assert len(X_train) > 0, 'insufficient number of samples'

print('BUILT: {} train time windows'.format(len(X_train)))

# BALANCE DATASET
print('2) Balancing data..')

X_train, y_train = utils.balance(X_train, y_train)

assert len(X_train) > 0, 'insufficient number of samples'

print('TRAIN: {} ↓time windows - {} ↑time windows'.format(len(np.where(y_train == 0)[0]), len(np.where(y_train)[0])))

# TRAIN MODEL
print('3) Training model..')

ssize = int(sqrt(timestep))
input_size = (ssize, ssize, len(features))

print('Timestep: {} - Futurestep: {} - Input size: {}'.format(timestep, futurestep, input_size))

model = utils.cnn(input_size)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)
hist = model.fit(X_train, y_train, shuffle=True, epochs=100, validation_split=0.2, callbacks=[es])

utils.plot_loss(hist)

# EVALUATE MODEL
print('4) Evaluating model..')

ohlcv_test, X_test, y_test = data_manager.build_windows(test_symbols)

print('TEST: {} ↓time windows - {} ↑time windows'.format(len(np.where(y_test == 0)[0]), len(np.where(y_test)[0])))

test_results = model.evaluate(X_test, y_test)
print('Test loss: {} - Test accuracy: {}'.format(test_results[0], test_results[1]))

# DISPLAY PREDICTIONS VS GT
print('5) Showing results for {} test symbols from sp500'.format(len(test_symbols)))

preds = model.predict_classes(X_test)
utils.plot_predictions(ohlcv_test, test_symbols, timestep, preds, y_test)

raw_input("Press Enter to exit..")
