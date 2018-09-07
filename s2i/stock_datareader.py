from datapackage import Package
import random
from datetime import datetime, timedelta
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web


__DP_URL = 'https://datahub.io/core/s-and-p-500-companies/datapackage.json'


def retrieve_sp500(limit=0, ratio=0):
    """
    Returns an array of S&P500 component stocks.
    If split_ratio is provided, the stocks are shuffled, divided into two disjoint sets and returned as a tuple.

    Args:
        limit (int): The number of S&P500 component stocks to retrieve (defaults to all).
        ratio (numeric): The ratio of the split between the S&P 500 component stocks.

    Returns:
        array or tuple: If ratio == 0 an array of str is returned.
                        If ratio != 0 a tuple of disjoint str sets is returned.

    Examples:
        >>>> len(stock_datareader.retrieve())
        505
        >>>> stock_datareader.retrieve(limit=5)
        ['AIZ', 'CTXS', 'PBCT', 'CSX', 'PVH']
        >>>> stock_datareader.retrieve(limit=5, ratio=.6)
        (['DISCA', 'VRTX', 'WEC'], ['CFG', 'BA'])
    """
    assert 0 <= ratio <= 1, 'invalid split ratio, must be in [0,1]'
    package = Package(__DP_URL)
    for resource in package.resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            symbols = [symbol[0].encode('utf-8') for symbol in resource.read()]
    if limit != 0:
        symbols = random.sample(symbols, limit)
    return symbols if ratio == 0 or ratio == 1 else _split(symbols, ratio)


def get_ohlcv(symbols):
    """
    Retrieve OHLCV data for the given symbols (data read from IEX).

    Args:
        symbols (object): The symbols for which to retrieve OHLCV information.

    Returns:
        dict: Key-value pair dictionary of OHLCV data.

    Examples:
        >>> [(k,v.size) for k, v in stock_datareader.get_ohlcv(['AAPL', 'MSFT']).iteritems()]
        [('AAPL', 6295), ('MSFT', 6295)]
        >>> [(k,v.columns.tolist()) for k, v in stock_datareader.get_ohlcv(['NFLX']).iteritems()]
        [('NFLX', [u'open', u'high', u'low', u'close', u'volume'])]
    """

    # cannot retrieve data for symbols unsupported by IEX
    symbols = _drop_unsupported_symbols(symbols)
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


def _split(symbols, ratio):
    split_idx = int(len(symbols) * ratio)
    return symbols[:split_idx], symbols[split_idx:]


def _drop_unsupported_symbols(symbols):
    supported_symbols = set([symbol.encode("utf-8") for symbol in web.get_iex_symbols()['symbol'].values])
    return list(set(symbols) & supported_symbols)

