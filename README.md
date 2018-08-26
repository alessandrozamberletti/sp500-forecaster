# stock2image
* __task:__ predict if the avg close price in the next 30 days will be higher/lower than the last 30 days avg close price;
* __input:__ high/low/close (hlc) data from the past 144 days. 

## Pipeline
1. gather S&P 500 hlc data
2. transform 144 days of normalized hlc data to a (12,12,3) image and feed it to a CNN.

   ![create-samples](./res/create-samples.gif)

* WIP WIP WIP
* model train
* img -> accuracy
* model eval
* img -> gt vs pred

## Dependencies
* tested on ubuntu 16.04, python 2.7
```
pip install -r requirements.txt
```
