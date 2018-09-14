# -*- coding: utf-8 -*-
import stock_utils
from stock_data_transformer import StockDataTransformer
from forecaster import Forecaster
from argparse import ArgumentParser
import logging
import numpy as np


def main(args):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('s2i')

    train_tickers, test_tickers = stock_utils.get_sp500_tickers(limit=args.stocknum, ratio=.8)
    transformer = StockDataTransformer(debug=args.debug)
    x, y = get_data(transformer, train_tickers)
    forecaster = Forecaster(transformer, debug=args.debug)
    forecaster.fit_to_data(x, y, epochs=args.epochs)
    for ticker in test_tickers:
        ohlcv = stock_utils.get_ohlcv(ticker)
        x, y = transformer.build_train_wins(ticker, ohlcv, balance=False)
        forecaster.evaluate(x, y, ohlcv, ticker)


def get_data(transformer, tickers):
    train_x = []
    train_y = []
    for ticker in tickers:
        ohlcv = stock_utils.get_ohlcv(ticker)
        x, y = transformer.build_train_wins(ticker, ohlcv, balance=True)
        train_x.append(x)
        train_y.append(y)
    return np.concatenate(train_x), np.concatenate(train_y)


def read_args():
    parser = ArgumentParser(description='Predict future stock trend.')
    parser.add_argument('stocknum', type=int, help='number of sp500 stocks to retrieve (0=all)')
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='show visual information')

    return parser.parse_args()


if __name__ == '__main__':
    main(read_args())
