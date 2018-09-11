# -*- coding: utf-8 -*-
import stock_utils
from stock_data_transformer import StockDataTransformer
from argparse import ArgumentParser
import os
import logging
import numpy as np

TIMESTEP = 144
FUTURESTEP = 30
FEATURES = ['high', 'low', 'close']
OUT_DIR = 'out'


def main(args):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('s2i')
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    train_tickers, _ = stock_utils.get_sp500_tickers(limit=args.stocknum, ratio=.8)
    transformer = StockDataTransformer(FEATURES, TIMESTEP, FUTURESTEP, debug=False)
    for ticker in train_tickers:
        ohlcv = stock_utils.get_ohlcv(ticker)
        log.info('%s - %i', ticker, ohlcv.size)
        x, y = transformer.get_train_windows(ticker, ohlcv, balance=False)
        log.info('↓%i - ↑%i - t%i', count_neg(y), count_pos(y), y.shape[0])
        last_x = transformer.get_latest_window(ticker, ohlcv)
        print last_x == x[-1, :]


def count_pos(windows):
    return np.count_nonzero(windows)


def count_neg(windows):
    return windows.shape[0] - np.count_nonzero(windows)


def read_args():
    parser = ArgumentParser(description='Predict future stock trend.')
    parser.add_argument('stocknum', type=int, help='number of sp500 stocks to retrieve (0=all)')
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='show visual information')

    return parser.parse_args()


if __name__ == '__main__':
    main(read_args())
