import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ohlc
# https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-importerror-cannot-import-name-is-list-like
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web

stocks = ['AMZN', 'AAPL']
data = web.DataReader(stocks, 'morningstar')

# for stock in stocks:
#     fig, ax = plt.subplots()
#     ohlc = data.xs(stock)
#     candlestick2_ohlc(ax, ohlc['Open'], ohlc['High'], ohlc['Low'], ohlc['Close'], width=0.6)
#     ax.set_title(stock)
#     plt.show()

import numpy as np
from sklearn.preprocessing import MinMaxScaler
ts = 14
X = []
y = []
for stock in stocks:
    ohlc = data.xs(stock).values
    for t in range(0, ohlc.shape[0] - ts):
        w = MinMaxScaler(feature_range=(0, 1)).fit_transform(ohlc[t:t+ts+1, 3:])
        current = w[:-1, :]
        future = w[-1, :]
        X.append(current)
        trending_up = future[0] > current[-1, 0]
        y.append(trending_up)

X = np.array(X)
y = np.array(y)

print X.shape
print y.shape
print np.count_nonzero(y)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten

model = Sequential()
model.add(LSTM(32, input_shape=(ts, 2), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, shuffle=True, epochs=100)
