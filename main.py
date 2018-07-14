import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, BatchNormalization, Convolution2D, MaxPooling2D, Activation
# https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-importerror-cannot-import-name-is-list-like
import pandas
pandas.core.common.is_list_like = pandas.api.types.is_list_like
import pandas_datareader as web
import cv2


def build_samples(stocks, timesteps):
    data = web.DataReader(stocks, 'morningstar')
    X = []
    y = []
    for stock in stocks:
        ohlc = data.xs(stock).values
        for t in range(0, ohlc.shape[0] - timesteps - 1):
            window = ohlc[t:t + timesteps, 1:4]
            future = ohlc[t + timesteps + 1, 1:4]
            y.append(future[-1] > window[-1, -1])

            p0 = window[0, :]
            window = [[(i[0]/p0[0])-1, (i[1]/p0[1])-1, (i[2]/p0[2])-1] for i in window]

            xx = np.array(window)
            xx = xx.reshape(12, 12, 3)
            # cv2.imshow('', xx)
            # cv2.waitKey(10)
            X.append(xx)

    return np.array(X), np.array(y)


ts = 144
X_train, y_train = build_samples(['AMZN',
                                  'AAPL',
                                  'NVDA',
                                  'BABA',
                                  'MSFT',
                                  'IBM',
                                  'CARB',
                                  'BIDU',
                                  'GDDY',
                                  'MOMO',
                                  'FB',
                                  'MMM',
                                  'ACN',
                                  'ADBE',
                                  'T'], ts)
X_test, y_test = build_samples(['NVDA'], ts)

model = Sequential()

model.add(Convolution2D(20, 3, 3, border_mode="same", input_shape=(12, 12, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print X_train.shape
model.fit(X_train, y_train, shuffle=True, epochs=100, validation_split=0.33)
