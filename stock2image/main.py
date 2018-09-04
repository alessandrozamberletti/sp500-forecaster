# -*- coding: utf-8 -*-
from symbol_manager import SymbolManager
from sp500 import SP500
import utils
from argparse import ArgumentParser
import os
import logging

TIMESTEP = 144
FUTURESTEP = 30
FEATURES = ['high', 'low', 'close']
OUT_DIR = 'out'


def main(args):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('stock2image')
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # RETRIEVE SYMBOLS
    log.info('Retrieving S&P500 data')
    train_symbols, test_symbols = SP500(limit=args.stocknum).split(.8)
    assert len(train_symbols) > 0 and len(test_symbols) > 0, 'no valid symbols found'
    log.info('%i train symbols - %i test symbols', len(train_symbols), len(test_symbols))

    # BUILD TIME WINDOWS
    log.info('Computing time windows')
    train_data = SymbolManager(train_symbols, FEATURES, TIMESTEP, FUTURESTEP, debug=args.debug).balance()
    assert len(train_data.x) == len(train_data.y) and len(train_data.x) > 0, 'insufficient number of samples'
    log.info('%i ↓time windows - %i ↑time windows', utils.count_neg(train_data.y), utils.count_pos(train_data.y))

    # TRAIN MODEL
    log.info('Training model')
    log.info('timestep: %i - futurestep: %i', TIMESTEP, FUTURESTEP)
    model, hist = utils.build_and_train_cnn(train_data, epochs=args.epochs)
    if args.debug:
        log.info('Saving loss plot to %s', OUT_DIR)
        utils.save_loss(hist)

    # EVALUATE MODEL
    log.info('Evaluating model')
    test_data = SymbolManager(test_symbols, FEATURES, TIMESTEP, FUTURESTEP, debug=args.debug)
    log.info('%i ↓time windows - %i ↑time windows', utils.count_neg(test_data.y), utils.count_pos(test_data.y))

    test_results = model.evaluate(test_data.x, test_data.y)
    log.info('test loss: %.2f - test accuracy: %.2f', test_results[0], test_results[1])

    log.info('Saving results to /%s for %i test symbols', OUT_DIR, len(test_symbols))
    preds = model.predict_classes(test_data.x)
    utils.save_predictions(test_data, preds)


def read_args():
    parser = ArgumentParser(description='Predict future stock trend.')
    parser.add_argument('stocknum', type=int, help='number of sp500 stocks to retrieve (0=all)')
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='show visual information')

    return parser.parse_args()


if __name__ == '__main__':
    main(read_args())
