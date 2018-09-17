# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from stock_forecaster import log, set_console_logger
from stock_forecaster.stock_data_transformer import StockDataTransformer
from stock_forecaster.forecaster import Forecaster
from stock_forecaster.utils import get_sp500_tickers, get_ohlcv, tickers2windows
from datetime import datetime
import os


def create(args):
    train_tickers, test_tickers = get_sp500_tickers(limit=args.stocknum, ratio=.8)
    log.debug("collected %i train, %i test sp500 stocks", len(train_tickers), len(test_tickers))

    log.debug('building train time windows')
    transformer = StockDataTransformer(debug=args.debug)
    x, y = tickers2windows(train_tickers, transformer)

    log.debug("training stock forecaster on %i time windows", x.shape[0])
    forecaster = Forecaster(transformer, debug=args.verbose, out_dir=args.output)
    hist = forecaster.fit_to_data(x, y, epochs=args.epochs)

    log.debug("evaluating forecaster on %i unseen stock/s", len(test_tickers))
    for idx, ticker in enumerate(test_tickers):
        ohlcv = get_ohlcv(ticker)
        x, y = transformer.build_train_windows(ticker, ohlcv, balance=False)
        oa, fn = forecaster.evaluate(x, y, ohlcv, ticker)
        log.debug("[%i - %s] OA: %.2f (%s)", idx, ticker, oa, fn)

    last_loss = hist.history['val_acc'][-1]
    fn = 'n{}_acc{:.2f}_{}.h5'.format(args.stocknum, last_loss, datetime.now().strftime("%Y%m%d%H%M%S"))
    model_path = os.path.join(args.output, fn)
    log.debug("saving trained stock forecaster to '%s'", model_path)
    forecaster.save(model_path)


def read_args():
    parser = ArgumentParser(description='Create future stock trend predictor.')
    parser.add_argument('stocknum', type=int, help='number of sp500 stocks to retrieve (0=all)')
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('-o', '--output', default='out', help='path to output directory (default ./out)')
    parser.add_argument('-v', '--verbose', action='store_true', default=True, help='verbose output, default true')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='visual train info, default false')

    return parser.parse_args()


if __name__ == '__main__':
    args = read_args()
    if args.verbose:
        set_console_logger()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    create(args)
