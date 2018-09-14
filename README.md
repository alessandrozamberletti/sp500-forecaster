# stock2image
* __task:__ predict if the next 30 days avg close price will be higher/lower than the past 30 days avg close price
* __input:__ high/low/close/volume (HLCV) data from the past 144 days

```console
usage: main.py [-h] [-d] stocknum epochs

Predict future stock trend.

positional arguments:
  stocknum     number of sp500 stocks to retrieve (0=all)
  epochs       number of training epochs

optional arguments:
  -h, --help   show this help message and exit
  -d, --debug  show visual information
```

## Pipeline
1. gather S&P500 HLC data and split into train/test
```console
INFO:stock2image:Retrieving S&P500 data
INFO:stock2image:404 train symbols - 101 test symbols
INFO:stock2image:Computing time windows
```

2. transform normalized OHLCV train data into train patches
<img src="./res/create.gif" width="500" height="375" />

3. train forecaster
<img src="./res/train.gif" width="500" height="375" />

4. evaluate against gt
<img src="./res/evaluate.gif" width="500" height="375" />

## Requirements
```console
az@ubuntu:~/stock2image$ cat /etc/issue*
Ubuntu 16.04.4 LTS
az@ubuntu:~/stock2image$ python --version
Python 2.7.12
az@ubuntu:~/stock2image$ pip install -r stock2image/requirements.txt
...
```
