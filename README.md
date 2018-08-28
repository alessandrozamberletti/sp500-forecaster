# stock2image
* __task:__ predict if the avg close price in the next 30 days will be higher/lower than the last 30 days avg close price
* __input:__ high/low/close (HLC) data from the past 144 days

## Pipeline
1. gather S&P500 HLC data and random split into train/test:
```console
* Retrieving S&P500 data..
** 404 train symbols - 101 test symbols
```

2. transform normalized HLC train data into (12,12,3) images:
<img src="./res/create-samples.gif" width="500" height="375" />

3. train model:
<img src="./res/train.gif" width="500" height="375" />

4. evaluate model:
* img -> gt vs pred

## Dependencies
* tested on ubuntu 16.04, python 2.7
```
pip install -r requirements.txt
```
