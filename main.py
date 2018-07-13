import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
# https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-importerror-cannot-import-name-is-list-like
import pandas
pandas.core.common.is_list_like = pandas.api.types.is_list_like
import pandas_datareader as web


def build_samples(stocks, timesteps):
    data = web.DataReader(stocks, 'morningstar')
    X = []
    y = []
    for stock in stocks:
        ohlc = data.xs(stock).values
        for t in range(0, ohlc.shape[0] - timesteps):
            scale = MinMaxScaler(feature_range=(0, 1)).fit(ohlc[:t + timesteps, 3:])
            w = scale.transform(ohlc[t:t + timesteps + 1, 3:])
            current = w[:-1, :]
            future = w[-1, :]
            X.append(current)
            y.append(future[0] > current[-1, 0])

    return np.array(X), np.array(y)


ts = 14
X_train, y_train = build_samples(['AMZN', 'AAPL'], ts)
X_test, y_test = build_samples(['NVDA'], ts)
print X_train.shape
print y_train.shape
print np.count_nonzero(y_train)

model = Sequential()
model.add(LSTM(32, input_shape=(ts, 2), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, shuffle=True, epochs=100)
print model.evaluate(X_test, y_test)
