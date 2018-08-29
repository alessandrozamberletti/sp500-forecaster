# stock2image
* __task:__ predict if the avg close price in the next 30 days will be higher/lower than the last 30 days avg close price
* __input:__ high/low/close (HLC) data from the past 144 days

## Pipeline
<details><summary>1. gather S&P500 HLC data and random split into train/test</summary>
<p>
  
```console
* Retrieving S&P500 data..
** 404 train symbols - 101 test symbols
```

</p>
</details>

<details><summary>2. transform normalized HLC train data into (12,12,3) images</summary>
<p>
  
<img src="./res/create.gif" width="500" height="375" />

</p>
</details>

<details><summary>3. train model</summary>
<p>
  
<img src="./res/train.gif" width="500" height="375" />

</p>
</details>

<details><summary>4. evaluate model</summary>
<p>
  
<img src="./res/evaluate.gif" width="500" height="375" />

</p>
</details>

## Dependencies
* tested on ubuntu 16.04, python 2.7
```
pip install -r requirements.txt
```
