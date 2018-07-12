import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ohlc
# https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-importerror-cannot-import-name-is-list-like
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web

stocks = ['AMZN', 'AAPL']
data = web.DataReader(stocks, 'morningstar')

for stock in stocks:
    fig, ax = plt.subplots()
    ohlc = data.xs(stock)
    candlestick2_ohlc(ax, ohlc['Open'], ohlc['High'], ohlc['Low'], ohlc['Close'], width=0.6)
    ax.set_title(stock)
    plt.show()
