import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
from datapackage import Package
import random

print 'Collecting SP500 stocks..'
package = Package('https://datahub.io/core/s-and-p-500-companies/datapackage.json')
for resource in package.resources:
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        sp500 = [s[0].encode('utf-8') for s in resource.read()]

stocks = random.sample(sp500, 25)
print('Retrieving data for: {}'.format(', '.join(stocks)))
timestep = 144
data = web.DataReader(stocks, data_source='morningstar')

print('Splitting into time samples..')
X = []
y = []
for stock in stocks:
    ohlc = data.xs(stock).values
    for t in range(0, ohlc.shape[0] - timestep - 30):
        current = ohlc[t:t + timestep, 1:4]
        future = ohlc[t + timestep:t + timestep + 30, 3]

        future_avg_price = np.average(future)
        current_avg_price = np.average(current[:, -1])
        y.append(future_avg_price > current_avg_price)

        p0 = current[0, :]
        current = [[(i[0]/p0[0])-1, (i[1]/p0[1])-1, (i[2]/p0[2])-1] for i in current]
        current = np.array(current).reshape(12, 12, 3)
        X.append(current)

print('{} time windows collected'.format(len(X)))

X = np.array(X)
y = np.array(y)
false_count = len(y) - np.count_nonzero(y)
true_idx, = np.where(y)

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
