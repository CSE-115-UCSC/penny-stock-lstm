import pandas as pd
import numpy as np
from polygon import RESTClient
import time
import datetime
from dateutil.relativedelta import relativedelta


'''Class gathering 2 years worth of historical stock data using Polygon API

Initialization Arguments: 
    - ticker: Stock ticker (string)
    - key: Polygon.io API Key (string)

Outputs:
    - ticker.csv: Generates CSV data file containing historical data

'''
class Aggregates():

    def __init__(self, ticker, key):
        self.client = RESTClient(api_key=key)

        self.ticker = ticker
        self.api_key = key

        self.start_date = "2022-01-09"
        self.end_date = "2024-01-09"
        self.multipler = 5
        self.timespan = "minute"


    '''Call Polygon API with desired timeframe & ticker specifications, store in Pandas DB

    Arguments: 
        - None

    Outputs:
        - ticker.csv: Generates CSV data file containing historical data

    '''
    def generate_historical(self):

        # initialize table
        historical_data = []

        # initialize sleeper to avoid API limits
        sleeper = 0 

        # create start timestamp
        start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")

        # loop until 2 years are exhausted
        while start < datetime.datetime.strptime(self.end_date, "%Y-%m-%d"):

            if (sleeper + 1) % 5 == 0:
                print("Sleeping to avoid breaching API limit")
                time.sleep(60)

            # create end timestamp (2 months ahead of start)
            end = start + relativedelta(months=2)

            # request and filter data
            print("Gathering data from", start.strftime("%Y-%m-%d"), "to", end.strftime("%Y-%m-%d"))
            
            # API call
            chunk_data = self.client.get_aggs(self.ticker, self.multipler, self.timespan, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            
            # remove non-standard hours
            filtered_data = self.filter_standard_hours(chunk_data)

            # add to database
            historical_data.extend(filtered_data)

            # move start ahead 2 months
            start = end

            # increment sleeper
            sleeper += 1
            if sleeper == 5:
                sleeper = 0

        # create dataframe to store data
        df = pd.DataFrame(historical_data, columns=['ticker', 't', 'o', 'h', 'l', 'c', 'v', 'n'])

        # save the DataFrame to a CSV file
        df.to_csv(f"{self.ticker}.csv", index=False)



    '''Filter out timestamps outside of 9:30 - 4:00 time range

    Arguments: 
        - data: Chunk of data from 2-year period

    Outputs:
        - filtered_data: Chunk of data from 2-year period with non-valid rows removed

    '''
    def filter_standard_hours(self, data):

        # upper and lower bounds for time of day
        day_start = datetime.time(9, 30)
        day_end = datetime.time(16, 0)

        # only select rows from data within specified range
        filtered_data = []
        for row in data:

            timestamp = datetime.datetime.fromtimestamp(row.timestamp / 1000).time()

            if day_start <= timestamp <= day_end:
                    
                    # extract features and create a dictionary
                    data_row = {
                        'ticker': self.ticker,
                        't': row.timestamp,
                        'o': row.open,
                        'h': row.high,
                        'l': row.low,
                        'c': row.close,
                        'v': row.volume,
                        'n': row.transactions
                    }
                    filtered_data.append(data_row)

        # return filtered data
        return filtered_data



if __name__ == "__main__":
    stock_data = Aggregates(ticker="Your Ticker Here", key="Your Key Here")
    stock_data.generate_historical()
