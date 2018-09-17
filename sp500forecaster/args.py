from argparse import ArgumentParser


def parse_create_args():
    parser = ArgumentParser(description='Create a model to predict future stock trends of SP500 companies.')
    parser.add_argument('stocknum', type=int, help='number of sp500 stocks to retrieve (0=all)')
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('-o', '--output', default='out', help='path to output directory (default ./out)')
    parser.add_argument('-v', '--verbose', action='store_true', default=True, help='verbose output, default true')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='visual train info, default false')

    return parser.parse_args()
