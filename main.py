import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
from datapackage import Package
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

# TODO: check how to speed it up, avoid locking for too long
print 'Collecting SP500 stocks..'
package = Package('https://datahub.io/core/s-and-p-500-companies/datapackage.json')
for resource in package.resources:
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        sp500 = [s[0].encode('utf-8') for s in resource.read()]

stocks = random.sample(sp500, 10)
print('Retrieving data for: {}'.format(', '.join(stocks)))
data = web.DataReader(stocks, data_source='morningstar')

# DROP WEEKENDS
data = data[data['Volume'] != 0]

# TODO: parametrize script
train_debug = False
timestep = 144
future_window = 30
ssize = int(sqrt(timestep))
chns = 3
print('timestep: {0} - future window: {1} - sample size: {2}x{2}x{3}'.format(timestep, future_window, ssize, chns))

# GATHER TRAIN SAMPLES
print('Splitting into time samples..')
X = []
y = []

if train_debug:
    plt.ion()
    f = plt.figure()
    gs = gridspec.GridSpec(3, 2)
    chart_ax = plt.subplot(gs[:, 0])
    visual_ax = []
    visual_ax_titles = ['Close', 'High', 'Low']
    for i in range(chns):
        visual_ax.append(plt.subplot(gs[i, -1]))

scaler = MinMaxScaler(feature_range=(0, 1))
for stock in stocks:
    ohlc = data.xs(stock).values
    for t in range(0, ohlc.shape[0] - timestep - future_window):
        current = ohlc[t:t + timestep, 1:4]
        future = ohlc[t + timestep - 1:t + timestep - 1 + future_window, 1:4]

        future_avg_price = np.average(future)
        current_avg_price = np.average(current[:, -1])
        trend = future_avg_price > current_avg_price

        p0 = current[0, :]
        current = np.array([[(i[0]/p0[0])-1, (i[1]/p0[1])-1, (i[2]/p0[2])-1] for i in current])
        future = np.array([[(i[0]/p0[0])-1, (i[1]/p0[1])-1, (i[2]/p0[2])-1] for i in future])

        current = scaler.fit_transform(current)
        future = scaler.transform(future)
        X.append(current.reshape(ssize, ssize, chns))
        y.append(trend)

        if train_debug:
            f.suptitle('Chart&Visual Train Samples - SYMBOL:{0}'.format(stock))

            chart_ax.cla()

            chart_ax.plot(current[:, -1])
            chart_ax.plot([np.average(current) for _ in range(timestep)], color='black', label='current avg price')

            xi = range(timestep - 1, timestep - 1 + future_window)
            color = 'green' if trend else 'red'
            chart_ax.plot(xi, future[:, -1], linestyle='--')
            chart_ax.plot(xi, [np.average(future) for _ in range(len(xi))], color=color, label='future avg price')

            # PRESENT|FUTURE LINE
            chart_ax.axvline(x=timestep - 1, color='gray', linestyle=':')

            chart_ax.set_title('Chart')
            chart_ax.set_xlabel('days')
            chart_ax.set_ylabel('normalized closing price')
            chart_ax.legend(loc='upper left')

            for i in range(len(visual_ax)):
                ax = visual_ax[i]
                ax.cla()
                ax.axis('off')
                ax.set_title(visual_ax_titles[i])
                ax.imshow(X[-1][:, :, i], cmap='gray')

            plt.show()
            plt.pause(.0001)

print('{} time windows collected'.format(len(X)))

# BALANCE DATA
X = np.array(X)
y = np.array(y)

false_windows = len(y) - np.count_nonzero(np.array(y))
true_idx, = np.where(y)
np.random.shuffle(true_idx)

print('{} downtrend and {} uptrend time windows'.format(false_windows, len(true_idx)))

X = np.delete(X, true_idx[false_windows:], axis=0)
y = np.delete(y, true_idx[false_windows:])

print('balanced to {} downtrend and {} uptrend time windows'.format(len(np.where(y == 0)[0]), len(np.where(y)[0])))

# TRAIN MODEL
model = Sequential()
model.add(Conv2D(20, (3, 3), padding="same", input_shape=(12, 12, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(20, (3, 3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, shuffle=True, epochs=100, validation_split=0.33)

# TODO: plot test data windows + classification vs expectation for random stocks
