# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from s2i import log, set_console_logger
from s2i.stock_data_transformer import StockDataTransformer
from s2i.forecaster import Forecaster
from s2i.stock_utils import get_sp500_tickers, get_ohlcv
import numpy as np


def main(args):
    if args.verbose:
        set_console_logger()

    train_tickers, test_tickers = get_sp500_tickers(limit=args.stocknum, ratio=.8)
    log.debug("COLLECTIONG SP500 STOCKS >> %i train, %i test", len(train_tickers), len(test_tickers))

    transformer = StockDataTransformer(debug=args.debug)
    x, y = get_data(transformer, train_tickers)
    forecaster = Forecaster(transformer, debug=args.verbose)

    log.debug("TRAINING STOCK FORECASTER >> %i train time windows", x.shape[0])
    forecaster.fit_to_data(x, y, epochs=args.epochs)

    for ticker in test_tickers:
        ohlcv = get_ohlcv(ticker)
        x, y = transformer.build_train_wins(ticker, ohlcv, balance=False)
        oa = forecaster.evaluate(x, y, ohlcv, ticker)
        log.debug("EVALUATION >> ticker: %s, oa: %.2f", ticker, oa)


def get_data(transformer, tickers):
    train_x = []
    train_y = []
    for ticker in tickers:
        ohlcv = get_ohlcv(ticker)
        x, y = transformer.build_train_wins(ticker, ohlcv, balance=True)
        train_x.append(x)
        train_y.append(y)
    return np.concatenate(train_x), np.concatenate(train_y)


def read_args():
    parser = ArgumentParser(description='Create future stock trend predictor.')
    parser.add_argument('stocknum', type=int, help='number of sp500 stocks to retrieve (0=all)')
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('-v', '--verbose', action='store_true', default=True, help='verbose output, default true')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='visual train info, default false')

    return parser.parse_args()


if __name__ == '__main__':
    main(read_args())
