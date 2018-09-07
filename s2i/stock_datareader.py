from datetime import datetime, timedelta
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web


def get_ohlcv(symbols):
    """
    Retrieve OHLCV data for the given symbols (data read from IEX).

    Args:
        symbols (str array): The symbols for which to retrieve OHLCV information.

    Returns:
        dict: Key-value pair dictionary of OHLCV data.

    Examples:
        >>> [(k,v.size) for k, v in stock_datareader.get_ohlcv(['AAPL', 'MSFT']).iteritems()]
        [('AAPL', 6295), ('MSFT', 6295)]
        >>> [(k,v.columns.tolist()) for k, v in stock_datareader.get_ohlcv(['NFLX']).iteritems()]
        [('NFLX', [u'open', u'high', u'low', u'close', u'volume'])]
    """

    # cannot retrieve data for symbols unsupported by IEX
    symbols = __drop_unsupported_symbols(symbols)
    assert len(symbols) > 0, 'none of the given symbols are supported by IEX stock exchange'

    # see: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#remote-data-iex
    start = datetime.now() - timedelta(days=2000)
    stock_data = {}
    for symbol in symbols:
        # NOTE: extracting too many symbols at once causes key error in IEXDailyReader
        # noinspection PyBroadException
        try:
            ohlcv = web.DataReader(symbol, data_source='iex', start=start)
        except Exception:
            print('no data for SYMBOL:{}, skipping'.format(symbol))
            continue

        stock_data[symbol] = ohlcv

    return stock_data


def __drop_unsupported_symbols(symbols):
    supported_symbols = set([symbol.encode("utf-8") for symbol in web.get_iex_symbols()['symbol'].values])
    return list(set(symbols) & supported_symbols)

