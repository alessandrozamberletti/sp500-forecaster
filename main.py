import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from datapackage import Package
from math import sqrt
from data_manager import DataManager

print 'Collecting SP500 stocks..'
package = Package('https://datahub.io/core/s-and-p-500-companies/datapackage.json')
for resource in package.resources:
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        sp500 = [s[0].encode('utf-8') for s in resource.read()]

# stocks = random.sample(sp500, 100)
stocks = sp500

split_pt = int(len(stocks)*.8)
train_stocks = stocks[:split_pt]
test_stocks = stocks[split_pt:]

print('{} symbols ({} train, {} test)'.format(len(stocks), len(train_stocks), len(test_stocks)))

timestep = 144
future_window = 30
ssize = int(sqrt(timestep))
features = ['open', 'high', 'low']
chns = len(features)

print('timestep: {0} - future window: {1} - sample size: {2}x{2}x{3}'.format(timestep, future_window, ssize, chns))

# COLLECT TIME WINDOWS
print('Splitting data into time windows..')
data_manager = DataManager(timestep, future_window, features, debug=False)
X_train, y_train = data_manager.build_windows(train_stocks)
X_test, y_test = data_manager.build_windows(test_stocks)

assert (len(X_train) == len(y_train)) and (len(X_test) == len(y_test)), 'non matching samples and targets lengths'
assert (len(X_train) > 0) and (len(X_test) > 0), 'insufficient number of samples'

print('{} train time windows collected'.format(len(X_train)))
print('{} test time windows collected'.format(len(X_test)))

# BALANCE DATA
downtrend_win_count = len(y_train) - np.count_nonzero(np.array(y_train))
uptrend_win_idx, = np.where(y_train)

print('{} downtrend and {} uptrend time windows before balancing'.format(downtrend_win_count, len(uptrend_win_idx)))

np.random.shuffle(uptrend_win_idx)
X_train = np.delete(X_train, uptrend_win_idx[downtrend_win_count:], axis=0)
y_train = np.delete(y_train, uptrend_win_idx[downtrend_win_count:])

assert len(X_train) == len(y_train), 'non matching samples and targets lengths'
assert len(X_train) > 0, 'insufficient number of samples'

print('{} downtrend and {} uptrend time windows after balancing'.format(len(np.where(y_train == 0)[0]), len(np.where(y_train)[0])))

exit(0)

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
model.fit(X_train, y_train, shuffle=True, epochs=10, validation_split=0.2)

# EVAL MODEL
preds = model.predict_classes(X_test)
# TODO: use zip
for i in range(len(preds)):
    pred = preds[i]
    out = 'OK' if y_test[i] == pred else 'KO'
    print('expected: {} vs. actual: {} -> {}'.format(y_test[i], bool(pred), out))

print('Test accuracy: {}'.format(model.evaluate(X_test, y_test)[1]))
