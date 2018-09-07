from datetime import datetime, timedelta
from tqdm import tqdm
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

    supported_symbols = set([symbol.encode("utf-8") for symbol in web.get_iex_symbols()['symbol'].values])
    stocks = list(set(stocks) & supported_symbols)
    assert len(stocks) > 0, 'none of the given symbols are supported by IEX stock exchange'

    # NOTE: data spans back to a maximum of 5 years
    start = datetime.now() - timedelta(days=2000)

    for symbol in tqdm(stocks, total=len(stocks)):
        # NOTE: extracting too many symbols at once causes key error in IEXDailyReader
        # noinspection PyBroadException
        try:
            ohlcv = web.DataReader(symbol, data_source='iex', start=start)
        except Exception:
            __skip(symbol, 'no data for SYMBOL:{}, skipping')
            continue

        stock_data[symbol] = ohlcv
    return stock_data


def __skip(self, symbol, message):
    print(message.format(symbol))
    self.symbols.remove(symbol)
