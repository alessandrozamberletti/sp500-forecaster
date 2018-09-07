from datetime import datetime, timedelta
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web


def get_ohlcv(stocks):
    """
    Returns a key-value pair dictionary of OHLCV data for the given stocks (data is read from IEX).

    :param stocks: array
        The stocks for which to retrieve OHLCV information.

    :return: Dictionary
    """
    stock_data = {}

    # cannot retrieve data for symbols unsupported by IEX
    stocks = __drop_unsupported_symbols(stocks)
    assert len(stocks) > 0, 'none of the given symbols are supported by IEX stock exchange'

    # see: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#remote-data-iex
    start = datetime.now() - timedelta(days=2000)

    for stock in stocks:
        # NOTE: extracting too many symbols at once causes key error in IEXDailyReader
        # noinspection PyBroadException
        try:
            ohlcv = web.DataReader(stock, data_source='iex', start=start)
        except Exception:
            print('no data for SYMBOL:{}, skipping'.format(stock))
            continue

        stock_data[stock] = ohlcv

    return stock_data


def __drop_unsupported_symbols(symbols):
    supported_symbols = set([symbol.encode("utf-8") for symbol in web.get_iex_symbols()['symbol'].values])
    return list(set(symbols) & supported_symbols)

