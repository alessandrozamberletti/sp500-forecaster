# sp500-forecaster
* __task:__ predict if the next 30 days avg close price will be higher/lower than the past 30 days avg close price
* __input:__ high/low/close/volume (HLCV) data from the past 144 days

```console
usage: create.py [-h] [-o OUTPUT] [-v] [-d] stocknum epochs

Create a model to predict future stock trends of SP500 companies.

positional arguments:
  stocknum              number of sp500 stocks to retrieve (0=all)
  epochs                number of training epochs

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT,            path to output directory (default ./out)
  -v, --verbose         verbose output, default true
  -d, --debug           visual train info, default false
```

## Pipeline
1. gather S&P500 HLC data and split into train/test
```console
sp500forecaster [DEBUG]: collected 404 train, 101 test sp500 stocks
sp500forecaster [DEBUG]: building train time windows
```

2. transform normalized OHLCV train data into train patches
<img src="./res/create.gif" width="500" height="375" />

3. train forecaster
<img src="./res/train.gif" width="500" height="375" />

4. evaluate against gt
<img src="./res/evaluate.gif" width="500" height="375" />

## Requirements
```console
az@ubuntu:~/sp500-forecaster$ cat /etc/issue*
Ubuntu 16.04.4 LTS
az@ubuntu:~/sp500-forecaster$ python --version
Python 2.7.12
az@ubuntu:~/sp500-forecaster$ pip install -r stock2image/requirements.txt
...
```
