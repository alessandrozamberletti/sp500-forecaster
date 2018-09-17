# input:
# ** symbol/s
# ** model path
# output:
# ** [predict_future(symbol) for symbol in symbols]
from sp500forecaster.args import parse_predict_args
from sp500forecaster import log, set_console_logger
import os


def predict(args):
    pass


if __name__ == '__main__':
    args = parse_predict_args()

    if args.verbose:
        set_console_logger()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    predict(args)
