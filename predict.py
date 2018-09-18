from sp500forecaster.args import parse_predict_args
from sp500forecaster import log, set_console_logger
from sp500forecaster.stock_data_transformer import StockDataTransformer
from sp500forecaster.forecaster import Forecaster
from sp500forecaster.utils import is_iex_supported, get_ohlcv
import os


def predict(args):
    symbols = [symbol.upper() for symbol in args.symbols]
    log.debug('symbols: %s', symbols)

    transformer = StockDataTransformer()
    forecaster = Forecaster(transformer)
    forecaster.load_weights(args.weights)

    [predict_future(symbol, transformer, forecaster) for symbol in symbols]


def predict_future(symbol, transformer, forecaster):
    if not is_iex_supported(symbol):
        log.debug('symbol %s is not supported by iex', symbol)

    log.debug('processing %s', symbol)
    last = transformer.build_latest_win(symbol, get_ohlcv(symbol))
    prediction = forecaster.predict_classes(last, batch_size=1)[0][0]
    status = 'positive' if prediction == 1 else 'negative'
    log.debug('%s future prediction for symbol %s', status, symbol)

    return prediction


if __name__ == '__main__':
    args = parse_predict_args()

    if args.verbose:
        set_console_logger()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    predict(args)
