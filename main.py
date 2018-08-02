import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from math import sqrt
from data_manager import DataManager
import random
import utils


# RETRIEVE SYMBOLS
print('0) Retrieving SP500 symbols..')

stocks = utils.sp500_symbols()

# TODO: remove sampling
stocks = random.sample(stocks, 10)

split_pt = int(len(stocks)*.8)
train_stocks = stocks[:split_pt]
test_stocks = stocks[split_pt:]

print('Found: {} train, {} test'.format(len(train_stocks), len(test_stocks)))

# BUILD TIME WINDOWS
print('1) Building time windows..')

timestep = 144
future_window = 30
features = ['open', 'high', 'low']
data_manager = DataManager(timestep, future_window, features, debug=False)
X_train, y_train = data_manager.build_windows(train_stocks)
X_test, y_test = data_manager.build_windows(test_stocks)

assert (len(X_train) > 0) and (len(X_test) > 0), 'insufficient number of samples'

print('Built: {} train, {} test'.format(len(X_train), len(X_test)))

# BALANCE DATASET
print('2) Balancing train data..')

downtrend_win_count = len(y_train) - np.count_nonzero(np.array(y_train))
uptrend_win_idx, = np.where(y_train)

print('Before: {} downtrend, {} uptrend'.format(downtrend_win_count, len(uptrend_win_idx)))

np.random.shuffle(uptrend_win_idx)
X_train = np.delete(X_train, uptrend_win_idx[downtrend_win_count:], axis=0)
y_train = np.delete(y_train, uptrend_win_idx[downtrend_win_count:])

assert len(X_train) == len(y_train), 'non matching samples and targets lengths'
assert len(X_train) > 0, 'insufficient number of samples'

print('After: {} downtrend, {} uptrend'.format(len(np.where(y_train == 0)[0]), len(np.where(y_train)[0])))

# TRAIN MODEL
print('3) Training model..')

ssize = int(sqrt(timestep))
chns = len(features)
print('Timestep: {0} - Futurestep: {1} - Input size: {2}x{2}x{3}'.format(timestep, future_window, ssize, chns))

model = Sequential()
model.add(Conv2D(20, (3, 3), padding="same", input_shape=(ssize, ssize, chns)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(20, (3, 3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(units=1, activation='sigmoid'))
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
