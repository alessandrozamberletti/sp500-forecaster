from datapackage import Package
import random
from datetime import datetime, timedelta
import warnings
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web


_DP_URL = 'https://datahub.io/core/s-and-p-500-companies/datapackage.json'


def get_sp500_tickers(limit=0, ratio=0):
    """
    Returns a list of S&P500 component tickers.
    If split_ratio is provided, the tickers are shuffled, divided into two disjoint lists and returned.

    Args:
        limit (int): The number of S&P500 component tickers to retrieve (defaults to all).
        ratio (numeric): The ratio of the split between the S&P 500 component tickers.

    Returns:
        list: If ratio == 0 a list of tickers is returned.
              If ratio != 0 two disjoint tickers lists are returned.

    Examples:
        >>>> len(stock_utils.get_sp500_tickers())
        505
        >>>> stock_utils.get_sp500_tickers(limit=5)
        ['AIZ', 'CTXS', 'PBCT', 'CSX', 'PVH']
        >>>> stock_utils.get_sp500_tickers(limit=5, ratio=.6)
        ['DISCA', 'VRTX', 'WEC'], ['CFG', 'BA']
    """
    assert 0 <= ratio <= 1, 'invalid split ratio, must be in [0,1]'
    package = Package(_DP_URL)
    for resource in package.resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            tickers = [ticker[0].encode('utf-8') for ticker in resource.read()]
    if limit != 0:
        tickers = random.sample(tickers, limit)
    return tickers if (ratio == 0 or ratio == 1) else _split(tickers, ratio)


def get_ohlcv(tickers):
    """
    Retrieve OHLCV data for the given tickers (data read from IEX).

    Args:
        tickers (list): The tickers for which to retrieve OHLCV information.

    Returns:
        dict: Key-value pair dictionary of OHLCV data.

    Examples:
        >>> [(k,v.size) for k, v in stock_utils.get_ohlcv(['AAPL', 'MSFT']).iteritems()]
        [('AAPL', 6295), ('MSFT', 6295)]
        >>> [(k,v.columns.tolist()) for k, v in stock_utils.get_ohlcv(['NFLX']).iteritems()]
        [('NFLX', [u'open', u'high', u'low', u'close', u'volume'])]
    """
    # cannot retrieve data for tickers unsupported by IEX
    tickers = _drop_unsupported_tickers(tickers)
    assert len(tickers) > 0, 'none of the given tickers are supported by IEX stock exchange'
    # see: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#remote-data-iex
    start = datetime.now() - timedelta(days=2000)
    stock_data = {}
    for ticker in tickers:
        # NOTE: extracting too many tickers at once causes key error in IEXDailyReader
        # noinspection PyBroadException
        try:
            ohlcv = web.DataReader(ticker, data_source='iex', start=start)
        except Exception:
            # ignore failed calls
            warnings.warn('no data for TICKER:{}, skipping'.format(ticker), Warning)
            continue
        stock_data[ticker] = ohlcv
    return stock_data


def _split(tickers, ratio):
    split_idx = int(len(tickers) * ratio)
    return tickers[:split_idx], tickers[split_idx:]


def _drop_unsupported_tickers(tickers):
    supported_tickers = set([ticker.encode("utf-8") for ticker in web.get_iex_symbols()['symbol'].values])
    print supported_tickers
    print tickers
    return list(set(tickers) & supported_tickers)
