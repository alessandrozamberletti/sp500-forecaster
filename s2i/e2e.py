# -*- coding: utf-8 -*-
import stock_utils
import nn_utils
from stock_data_transformer import StockDataTransformer
from argparse import ArgumentParser
import logging
import numpy as np

TIMESTEP = 144
FUTURESTEP = 30
FEATURES = ['high', 'low', 'close']


def main(args):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('s2i')

    train_tickers, test_tickers = stock_utils.get_sp500_tickers(limit=args.stocknum, ratio=.8)
    transformer = StockDataTransformer(FEATURES, TIMESTEP, FUTURESTEP, debug=False)
    train_x = []
    train_y = []
    for ticker in train_tickers:
        ohlcv = stock_utils.get_ohlcv(ticker)
        log.info('%s - %i', ticker, ohlcv.size)
        x, y = transformer.build_train_wins(ticker, ohlcv, balance=True)
        train_x.append(x)
        train_y.append(y)
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    model, _ = nn_utils.build_and_train_cnn(train_x, train_y, epochs=args.epochs, save_loss=True)


def read_args():
    parser = ArgumentParser(description='Predict future stock trend.')
    parser.add_argument('stocknum', type=int, help='number of sp500 stocks to retrieve (0=all)')
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='show visual information')

    return parser.parse_args()


if __name__ == '__main__':
    main(read_args())
