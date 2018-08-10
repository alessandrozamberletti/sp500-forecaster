# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from symbol_manager import SymbolManager
import random
import utils

timestep = 144
futurestep = 30
debug = False
features = ['high', 'low', 'close']

# RETRIEVE SYMBOLS
print('* Retrieving S&P500 data..')
sp500_symbols = utils.sp500_symbols()
sp500_symbols = random.sample(sp500_symbols, 3)
train_symbols, test_symbols = utils.split(sp500_symbols, .8)
assert len(train_symbols) > 0 and len(test_symbols) > 0, 'no valid symbols found'
print('** {} train symbols - {} test symbols'.format(len(train_symbols), len(test_symbols)))

# BUILD TIME WINDOWS
print('* Computing time windows..')
symbol_manager = SymbolManager(features, debug=debug)
xy_train = symbol_manager.build_windows(train_symbols, timestep, futurestep)
assert len(xy_train) > 0, 'insufficient number of samples'

# BALANCE DATASET
print('* Balancing data..')
X_train, y_train = utils.balance(utils.vectorize(xy_train, 'x'), utils.vectorize(xy_train, 'y'))
assert len(X_train) > 0, 'insufficient number of samples'
print('** {} ↓time windows - {} ↑time windows'.format(len(np.where(y_train == 0)[0]), len(np.where(y_train)[0])))

# TRAIN MODEL
print('* Training model..')
ssize = int(sqrt(timestep))
input_shape = (ssize, ssize, len(features))
print('** timestep: {} - futurestep: {} - model input shape: {}'.format(timestep, futurestep, input_shape))

model, hist = utils.cnn(X_train, y_train, input_shape)
utils.plot_loss(hist)

# EVALUATE MODEL
print('* Evaluating model..')
xy_test = symbol_manager.build_windows(test_symbols, timestep, futurestep)
X_test = utils.vectorize(xy_test, 'x')
y_test = utils.vectorize(xy_test, 'y')
print('** {} ↓time windows - {} ↑time windows'.format(len(np.where(y_test == 0)[0]), len(np.where(y_test)[0])))

test_results = model.evaluate(X_test, y_test)
print('** test loss: {} - test accuracy: {}'.format(test_results[0], test_results[1]))

print('* Saving results to /out for {} test symbols'.format(len(test_symbols)))
preds = model.predict_classes(X_test)
utils.save_predictions(xy_test, timestep, futurestep, preds)
