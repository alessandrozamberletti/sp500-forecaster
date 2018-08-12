# stock2image

* For any stock, given hlc data from the past 144 days, predict if the average close price in the next 30 days will be
higher or lower than the last 30 days average close price. 
* Instead of using rnn/lstm, we transform hlc data to a (12,12,3) image and feed it to a cnn.

## Pipeline
* From stock data to image:
![create-samples](./res/create-samples.gif)
* model train
* img -> accuracy
* model eval
* img -> gt vs pred

* tested on ubuntu 16.04, python 2.7
```
pip install -r requirements.txt
```
