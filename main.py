# -*- coding: utf-8 -*-
from symbol_manager import SymbolManager
from sp500 import SP500
import utils

timestep = 144
futurestep = 30
debug = False
features = ['high', 'low', 'close']

# RETRIEVE SYMBOLS
print('* Retrieving S&P500 data..')
train_symbols, test_symbols = SP500(limit=10).split(.8)
assert len(train_symbols) > 0 and len(test_symbols) > 0, 'no valid symbols found'
print('** {} train symbols - {} test symbols'.format(len(train_symbols), len(test_symbols)))

# BUILD TIME WINDOWS
print('* Computing time windows..')
train_data = SymbolManager(train_symbols, features, timestep, futurestep, debug=debug).balance()
assert len(train_data.x) == len(train_data.y) and len(train_data.x) > 0, 'insufficient number of samples'
print('** {} ↓time windows - {} ↑time windows'.format(utils.count_neg(train_data.y), utils.count_pos(train_data.y)))

# TRAIN MODEL
print('* Training model..')
print('** timestep: {} - futurestep: {}'.format(timestep, futurestep))
model, hist = utils.build_and_train_cnn(train_data, epochs=10)
utils.plot_loss(hist)

# EVALUATE MODEL
print('* Evaluating model..')
test_data = SymbolManager(test_symbols, features, timestep, futurestep, debug=debug)
print('** {} ↓time windows - {} ↑time windows'.format(utils.count_neg(test_data.y), utils.count_pos(test_data.y)))

test_results = model.evaluate(test_data.x, test_data.y)
print('** test loss: {} - test accuracy: {}'.format(test_results[0], test_results[1]))

print('* Saving results to /out for {} test symbols'.format(len(test_symbols)))
preds = model.predict_classes(test_data.x)
utils.save_predictions(test_data, preds)
