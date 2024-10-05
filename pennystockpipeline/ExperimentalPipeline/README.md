Recently, as I've been experimenting with different data sources, I've put together an additional timeline to streamline gathering/processing stock data. Here is most everything I think you would need to know to use it, debug it, or make changes to it to suit your needs.

There are several key parts to the pipeline, split amongst various files:
1. **create_database.ipynb** -  *Deals with compiling together a list of tickers, gathering historical data, imputing the data, and saving to a local database file.*
    1. **StockData.py** - *Classes/methods to support create_database.ipynb*
3. **model_training.ipynb**
    1. **Preprocessor.py** - *Classes/methods to support model_training.ipynb*
    2. **Model.py** - *Classes/methods to support model_training.ipynb*


---
# create_database.ipynb

To begin, **create_database.ipynb** is split into several parts. At the very top you'll find a place for your API keys, but at the moment, functionality with Polygon.io for [historical data](#historical_data) and Finnhub for [stock quotes](#quote) are supported.
```
import StockData
polygon_key = ""
finnhub_key = ""
alpha_vantage_key = ""
```

## Gathering list of tickers
### StockData.py/Tickers(StockData)

To gather stock tickers efficiently, there is class StockData with an associated method Tickers(). An example of its use appears as such:
```
penny_stock_tickers = StockData.Tickers(polygon_key=polygon_key, 
                    finnhub_key=finnhub_key,
                    market='otc', 
                    stock_type='penny', 
                    rank_by_volume=True,
)

penny_stock_tickers.get_tickers(amount=50)
```
There are several parameters:
1. **finhub_key**: 
    - Used to gather a **quote** (see rank_by_volume() below, most recent close price of the stock), which is used in the method: *Stockdata.remove_non_penny_stocks()* to remove tickers with a most recent close price greater than a specified value.
    - This value is modifiable and can be found within Stockdata.py -> Stockdata() -> remove_non_penny_stocks -> "if last_close >5".
2. **market**: All of the tickers are pulled from Polygon.io's database, and they have several thousand tickers from many markets. For OTC, use 'otc', and for NASDAQ, use 'stocks.'
3. **stock_type**: Setting *stock_type='penny'* will trigger the function call self.remove_non_penny_stocks() during the ticker-gathering phase. Setting the type to anything else will ignore that function call.
4. **rank_by_volume**: Gathering tickers from Polygon.io's database is a lengthy process, and I put this flag here with the intention of allowing you to grab the most 'active' stocks quickly. The reason it takes so long to gather tickers is because it must follow this process:
    1. Call Polygon.io to get a list of **all** tickers (Around 20,000). This is a one-time API call which is instantaneous, but lists all penny and non-penny stocks of the specified market.
    2. For each of these tickers, use Finnhub's API to get a **quote** of each ticker to discard/keep each ticker depending on whether or not it can be classified as a penny stock. Finnhub restricts to 60 API calls per minute, so getting a list of more than 60 tickers will take a small but non-zero amount of time. 
    3. With the remaining penny stock tickers, which could be in the tens of thousands if you set the request amount that high, sort_by_volume() will do as such so that you don't have to waste time gathering historical data on tickers that have very spotty data.
    
## Gathering historical data
### StockData.py/Historical(StockData)

Once an instance of **StockData.tickers** is created and the method *penny_stock_tickers.get_tickers(amount)* is ran, the list of all tickers will be stored in:
```
penny_stock_tickers.tickers
```
and can be viewed with:
```
print(penny_stock_tickers)
```
*Note: I've tried Finra, Finnhub, AlphaVantage, TwelveData, etc., but with little luck. Though I've spent a lot of time trying to get them to work, I've met various barries of spotty data/premium plans, but even still, I don't think I've done the best that can be done. I encourage you to experiment with other APIs if possible)*


### Gathering
Now, to gather historical data from Polygon.io's records, call the function
```
penny_stock_historical_data = StockData.Historical(
        api="polygon", 
        api_key=polygon_key, 
        tickers=penny_stock_tickers.tickers, 
        start_date='2023-01-01', end_date='2024-01-01'
)
```
There are several parameters to allow you to customize the data you will receive:
1. **API**: Right now, only Polygon is functional
2. **api_key**: Specified at top of notebook, please pass your Polygon key through and remember to not push it to Github.
3. **tickers**: This takes in the list of tickers to gather data for, which is stored in *penny_stock_tickers.tickers* from the previous function. You can modify this parameter if you wish to gather data for something else quickly and on the fly, just be sure to instantiate a **penny_stock_historical** object first so that the method StockData.Historical() may be called on it.
4. **start_date**: I've found that with Polygon.io's API, the free plan only gives us access to data from 2023-01-01 to the present, but if you wish to alter this range, change these parameters. 

Specifically, the data gathers is from the hours 9:30 - 16:00 at intervals of 5 minutes. If you wish to change this, look to:

*StockData.py -> Historical(StockData) -> def gather_historical(self, extra_hours=False) ->  market_open/market_close*
Additionally, a flag 
```
after_hours=False
```
can be passed to gather data in hours outside of 9:30 - 16:00, but I advise against this as sequence lengths should be the same. 

## Saving the data
**Imporant Note**: It should be noted that the above function will look for a director called 
```
~historical_data/
```
where it will output each {ticker}.csv file. Once this is complete, the function
```
penny_stock_historical_data.to_sql()
```
is called to combine each individual {ticker}.csv into a single **stockdata.csv** and a corresponding **stockdata.db**, both also within the ~/historical_data directories.

## Imputing the data
To help ensure effective training, I've added an imputation function with a variety of customizable parameters for ease of use. Observe the function:
```
penny_stock_historical_data.impute(
        amount="day", 
        fill="forward", 
        dropout_threshold=0.65)
```

1. **amount**: This parameter can take two forms, each with their own pros/cons, though I recommend "day".
    1. **day**: The function will comb over the entire joined dataset and look for every day that has at least one data-point. For each of those days, data is imputed from 9:30 - 16:00 **of that day only**. In turn, we are left with less fabricated data in the end, as days that initially had 0 data still have 0 data. It is important to know that using "day" will result in a smaller **ratio** (see below) on average for each stock, as it "plays to each of their strengths".
    2. **complete**: The function will comb over the entire joined dataset and impute-to-fill **every day** for the entire historical period (year). In turn, every stock has the same *amount* of data (by rows), but stocks that are more spotty will have more fabricated than actual data.
2. **fill**: Different methods of imputation are available, though, at the moment, forward is the only one I have working (interpolate_linear is close but I'm still getting it to work properly with all columns). Ideally, there will be several methods (not comprehensive):
    1. **forward**: Copies the next known data point forward until next real data point to fill the gaps, but leads to unnatural jumps and flat zones.
    2. **interpolate_linear**: Connects both sides of each gaps with a linear sequence of datapoints.
    3. **interpolate_polynomial**: More realistic than linear as they are connected with a polynomial curve
    4. **rolling**: Similar to forward, but copies the last known data point. Will be implemented soon.
3. **dropout_threshold**: I would argue this is the most important parameter to monitor. Within the ```Historical.impute_data()```, I store a  **ratio** for each stock, which helps determine whether or not the data is usable. Prior to imputation, an ```original_length = len(temp_data)``` is stored, and after imputation, this is used in the line: ```ratio = original_length / len(temp_imputed)```. This value is printed during the imputation process and helps us visualize how much real vs fabricated data we are left with for each stock following imputation. 
For example, if prior to imputation, a stock has 10 rows of data, but after imputation, it has 100, we would have a ratio of ```0.1```. This tells us that the stock data is mostly fabricated and therefore may lead to overfitting contamination. Conversely, a stock with 95 rows prior and 100 rows post will have a ratio of ```0.95```. By setting ```dropout_threshold=0.7``` for instance, we would remove all stocks with ratios less than 0.7 from our imputed database.

After ```penny_stock_historical.impute()``` is called, you will find new files
1. stockdata_imputed.db
2. stockdata_imputed.csv

within your ~/historical_data/ directory.

---
# model_training.ipynb

With the imputed dataset now formed, we will perform several pre-processing methods on it to prepare for use in training a model. Please note that the methods used will now vary depending on what model you intend to use. For the methods listed below, they are currently intended for LSTM use.

## Preprocessing: Additional Features

```
import StockData
import Preprocessor
import Model
```

```
data_preprocessor = Preprocessor.Preprocessor(
        'historical_data/stockdata_imputed.db',
        drift=True, 
        volatility=True, 
        moving_avg=True, 
        momentum=True, 
        volume_features=True,
        support_resistance=True,
        bollinger_bands=True, 
        z_score=True)
```
Using the initial columns we obtained from Polygon.io
| pst | close | high | low | open | otc  | ticker | timestamp | transactions | volume | vwap |
|---|---|---|---|---|---|---|---|---|---|---|
| 1/3/23 9:30 | 45.11 | 45.12 | 44.9 | 44.9 |   | AA | 1.67277E+12 | 371 | 19714 | 44.994 |

We can optionally derive new features such as drift, volatility, moving_avg, momentum, and more to describe trends in the data. They likely will not be used by all models, so please toggle them on or off depending on your specific needs.
Regardless of if/how the columns are changed, 2 new files will be added to your ~/historical_data directory to reflect:
1. sd_pre.db
2. sd_pre.csv

**Note:** All of the methods used to add new columns can be found in **Preprocessor.py**.

## Preprocessing (Sequencing & Normalization) -> Training 

With out last changes to the database being made, we are ready to prepare the data to be fed into our mode. By calling
```
model = Model.StockLSTMModel('historical_data/sd_pre.db')
```
we are creating an instance of our LSTM model based on the dataframe stored in the file "sd_pre.db".

From here, we can call
```
model.train_model()
```
on our model object, which calls several functions under the hood from our **Model.py**. This module is still a work in process, but these are the methods so far (to limited working capability):
1. **__init__**: Using the passed in filepath to the .db, stores it within the objects self-properties.
2. **train_model(self):** Performs these actions (Will describe more detail TBA)
    1. Appoints self.df = **load_data()**
    2. Appoints X, y = **self.prepare_data(self.df)**
    3. Reshapes the data
    3. Appoints self.model = **self.create_model((X.shape[1], 1))**
    4. Fits the model and saves weights as "lstm_model.h5" to local directory
3. **load_data(self):** With the stored filepath, opens the db and saves the list of unique tickers to **self.tickers** and returns the dataframe.
3. **prepare_data(self, df, time_step=78):** Scales each feature of each stock separately and returns sequenced scaled data.
5. **create_model(self, input_shape:**: Creates a standard Tensorflow LSTM Sequential model, hyperparameters and layers customizeable here. Returns model.


## Predicting

WIP
Once the model is finished training, run predictions on new stock data (imputed and preprocessed the same as training data) and plot predictions. Compare with other models


