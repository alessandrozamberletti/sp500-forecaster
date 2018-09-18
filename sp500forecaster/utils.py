from datapackage import Package
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web


_DP_URL = 'https://datahub.io/core/s-and-p-500-companies/datapackage.json'
_IEX_TICKERS = set([ticker.encode("utf-8") for ticker in web.get_iex_symbols()['symbol'].values])


def get_sp500_tickers(limit=0, ratio=0, iex_supported=True):
    """
    Returns a list of S&P500 component tickers.
    If split_ratio is provided, the tickers are shuffled, divided into two disjoint lists and returned.

    Args:
        limit (int): The number of S&P500 component tickers to retrieve (defaults to all).
        ratio (numeric): The ratio of the split between the S&P 500 component tickers.
        iex_supported (bool): If True then only iex-supported SP500 symbols are returned (default True).

    Returns:
        list: If ratio == 0 a list of tickers is returned.
              If ratio != 0 two disjoint tickers lists are returned.

    Examples:
        >>>> len(utils.get_sp500_tickers())
        505
        >>>> utils.get_sp500_tickers(limit=5)
        ['AIZ', 'CTXS', 'PBCT', 'CSX', 'PVH']
        >>>> utils.get_sp500_tickers(limit=5, ratio=.6)
        ['DISCA', 'VRTX', 'WEC'], ['CFG', 'BA']
    """
    assert 0 <= ratio <= 1, 'invalid split ratio, must be in [0,1]'
    package = Package(_DP_URL)
    for resource in package.resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            tickers = [ticker[0].encode('utf-8') for ticker in resource.read()]
    if iex_supported:
        tickers = list(set(tickers) & _IEX_TICKERS)
    if limit != 0:
        tickers = random.sample(tickers, limit)
    return tickers if (ratio == 0 or ratio == 1) else _split(tickers, ratio)


def get_ohlcv(ticker):
    """
    Retrieve OHLCV data for the given ticker (data read from IEX).

    Args:
        ticker (str): The ticker for which to retrieve OHLCV information.

    Returns:
        dataframe: Pandas dataframe of OHLCV data.

    Examples:
        >>> print utils.get_ohlcv('NFLX').columns.tolist()
        [u'open', u'high', u'low', u'close', u'volume']
    """
    # cannot retrieve data for tickers unsupported by IEX
    if not is_iex_supported(ticker):
        raise ValueError('{} is not supported by IEX stock exchange'.format(ticker))
    # see: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#remote-data-iex
    start = datetime.now() - timedelta(days=2000)
    return web.DataReader(ticker, data_source='iex', start=start)


def tickers2windows(tickers, transformer):
    """
    Transform a list of iex-supported stock symbols into a train dataset using the provided transformer.

    Args:
        tickers (list): a list of iex-supported symbols (see get_sp500_tickers function).
        transformer (object): A StockDataTransformer object, used to convert ohlcv data to time windows.

    Returns:
        np.array, np.array: Time windows with their respective ground-truths (False=neg. forecast, True=pos. forecast).

    Examples:
        >>>> from sp500forecaster.stock_data_transformer import StockDataTransformer
        >>>> x, y = utils.tickers2windows(['AAPL'], StockDataTransformer())
        >>>> x.shape[0] == y.shape[0]
        True
        >>>> import numpy as np
        >>>> np.unique(y)
        array([False,  True])
    """
    train_x = []
    train_y = []
    for ticker in tickers:
        ohlcv = get_ohlcv(ticker)
        x, y = transformer.build_train_windows(ticker, ohlcv, balance=True)
        train_x.append(x)
        train_y.append(y)
    return np.concatenate(train_x), np.concatenate(train_y)


def is_iex_supported(ticker):
    """
    Check whether a ticker is supported by iex.

    Args:
        ticker (str): stock ticker.

    Returns:
        bool: True if ticker is supported by iex, False otherwise.

    Examples:
        >>>> utils.is_supported('AAPL')
        True
    """
    return ticker in _IEX_TICKERS


def _split(tickers, ratio):
    split_idx = int(len(tickers) * ratio)
    return tickers[:split_idx], tickers[split_idx:]
