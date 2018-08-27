# stock2image
* __task:__ predict if the avg close price in the next 30 days will be higher/lower than the last 30 days avg close price;
* __input:__ high/low/close (hlc) data from the past 144 days. 

## Pipeline
1. gather S&P 500 hlc data;
2. transform 144 days of normalized hlc data to (12,12,3) images;

   ![create-samples](./res/create-samples.gif)

3. train CNN on train data from (2);

* img -> accuracy

* WIP WIP WIP
* model eval
* img -> gt vs pred

## Dependencies
* tested on ubuntu 16.04, python 2.7
```
pip install -r requirements.txt
```
