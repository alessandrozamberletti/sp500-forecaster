import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
from datapackage import Package
import random
from math import sqrt
from data_manager import DataManager

print 'Collecting SP500 stocks..'
package = Package('https://datahub.io/core/s-and-p-500-companies/datapackage.json')
for resource in package.resources:
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        sp500 = [s[0].encode('utf-8') for s in resource.read()]

stocks_num = 1
stocks = random.sample(sp500, stocks_num)

# TODO: evaluate on different stocks to avoid fake results caused by similar time windows between train and test
# SPLIT
split_pt = int(len(stocks)*.8)
train_stocks = stocks[:split_pt]
test_stocks = stocks[split_pt:]

# TODO: morningstar route returns 404, find sth else
print('Collecting data for: {}'.format(', '.join(stocks)))
data = web.DataReader(stocks, data_source='morningstar', retry_count=0)

# DROP NaN and WEEKENDS
data = data.dropna()
data = data[data['Volume'] != 0]

# TODO: parametrize script
timestep = 144
future_window = 30
ssize = int(sqrt(timestep))
chns = 3
print('timestep: {0} - future window: {1} - sample size: {2}x{2}x{3}'.format(timestep, future_window, ssize, chns))

data_manager = DataManager(timestep, future_window, chns)

# GATHER TRAIN SAMPLES
print('Splitting data into time windows..')
X = []
y = []
X_test = []
y_test = []

for stock in stocks:
    try:
        ohlc = data.xs(stock).values
    except KeyError:
        continue

    xs, ys = data_manager.build_samples(stock, ohlc)
    if stock in test_stocks:
        X_test.append(xs)
        y_test.append(ys)
    else:
        X.append(xs)
        y.append(ys)

print('{} time windows collected'.format(len(X)))

# BALANCE DATA
downtrend_win_count = len(y) - np.count_nonzero(np.array(y))
uptrend_win_idx, = np.where(y)

print('{} downtrend and {} uptrend time windows before balancing'.format(downtrend_win_count, len(uptrend_win_idx)))

np.random.shuffle(uptrend_win_idx)
X = np.delete(X, uptrend_win_idx[downtrend_win_count:], axis=0)
y = np.delete(y, uptrend_win_idx[downtrend_win_count:])

print('{} downtrend and {} uptrend time windows after balancing'.format(len(np.where(y == 0)[0]), len(np.where(y)[0])))

# TRAIN MODEL
model = Sequential()
model.add(Conv2D(20, (3, 3), padding="same", input_shape=(ssize, ssize, chns)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(20, (3, 3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, shuffle=True, epochs=10, validation_split=0.2)

# EVAL MODEL
preds = model.predict_classes(X_test)
for i in range(len(preds)):
    pred = preds[i]
    out = 'OK' if y_test[i] == pred else 'KO'
    print('expected: {} vs. actual: {} -> {}'.format(y_test[i], bool(pred), out))

print('Test accuracy: {}'.format(model.evaluate(X_test, y_test)[1]))
