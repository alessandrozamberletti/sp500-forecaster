from datapackage import Package
import random


__DP_URL = 'https://datahub.io/core/s-and-p-500-companies/datapackage.json'


def retrieve(limit=0, ratio=0):
    """
    Returns an array of S&P500 component stocks.
    If split_ratio is provided, the stocks are divided into two disjoint sets and returned as a tuple.

    :param limit: {int, 0}
        The number of S&P500 component stocks to retrieve (defaults to all).
    :param ratio: {numeric, 0}
        The ratio of the split between the S&P 500 component stocks.

    :return: Array or tuple
    """
    assert 0 <= ratio <= 1, 'invalid split ratio, must be in [0,1]'
    package = Package(__DP_URL)
    for resource in package.resources:
        if resource.descriptor['datahub']['type'] == 'derived/csv':
            symbols = [symbol[0].encode('utf-8') for symbol in resource.read()]
    if limit != 0:
        symbols = random.sample(symbols, limit)
    return symbols if ratio == 0 or ratio == 1 else __split(symbols, ratio)


def __split(symbols, ratio):
    split_idx = int(len(symbols) * ratio)
    return symbols[:split_idx], symbols[split_idx:]
