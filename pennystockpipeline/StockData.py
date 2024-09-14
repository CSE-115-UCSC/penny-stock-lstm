import pandas as pd
import polygon
import finnhub
from alpha_vantage.timeseries import TimeSeries
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import sqlite3


# Class Stock_Data - creates sql database
# Subclass Tickers - gets tickers
# Subclass Historical - gets data

'''Class containing stock database

Initialization Arguments: 
    - Polygon.io API Key

Representable Form:
    - 

'''
class StockData():
    
    def __init__(self, polygon_key):
        self.polygon_client= polygon.RESTClient(api_key=polygon_key)
        self.polygon_key = polygon_key
        


'''Subclass containing list of otc penny stock tickers using Polygon.io API

Initialization Arguments: 
    - Polygon.io API Key
    - Finnhub.io API Key
    - market_type (ex: otc, NASDAQ)
    - stock_type (ex: penny, any)

Representable Form:
    - self.tickers: List of all tickers (ex: ['AAAIF', 'AABB', 'AABVF'])

'''
class Tickers(StockData):

    def __init__(self, polygon_key, finnhub_key, market, stock_type):
        super().__init__(polygon_key)
        self.finnhub_client = finnhub.Client(api_key=finnhub_key)
        self.finnhub_key = finnhub_key
        self.market = market
        self.stock_type = stock_type
        self.tickers=[]

    def __repr__(self):
        return str(self.tickers)
    
    def get_tickers(self, amount=30):

        print(f"Gathering {amount} tickers from the {self.market} market. . .")

        tickers_gathered = 0

        # Loop to gather 'amount' number of tickers
        while tickers_gathered < amount:

            # Call Polygon.io API based on specifications
            try: 
                page_of_tickers = self.polygon_client.list_tickers(
                    market=self.market,
                    active=True,
                    sort='ticker',
                    order='asc',
                    limit=amount
                )
            except Exception as e:
                sys.stderr.write(f"Failed to make list_tickers API call: {e}")
                break

            # Comb through API page result and append all tickers to a list
            for ticker in page_of_tickers:
                if tickers_gathered >= amount:
                    break
                self.tickers.append(ticker.ticker)
                tickers_gathered += 1

        # Remove non-penny stocks if necessary
        if self.stock_type == 'penny':
            self.remove_non_penny_stocks()

        print(f"Returned {len(self.tickers)} valid tickers:\n{self.tickers}")

    def remove_non_penny_stocks(self):

        # Remove all non-penny stocks from self.tickers
        for ticker in self.tickers:
            
            try:
                last_close = self.finnhub_client.quote(ticker)['c']
                if last_close > 5:
                    self.tickers.remove(ticker)
            
            except Exception as e:
                sys.stderr.write(f"Failed to make get_last_trade API call: {e}")
                break

    def remove_outdated_stocks(self):
        pass


'''Subclass containing list of otc penny stock tickers using Polygon.io API

Initialization Arguments: 
    - Polygon.io API Key
    - start_date (ex: 2022-01-01)
    - end_data (ex: 2024-01-01)

Representable Form:
    - self.tickers: List of all tickers (ex: ['AAAIF', 'AABB', 'AABVF'])

'''
class Historical(StockData):

    def __init__(self, api, api_key, tickers, start_date, end_date):
        super().__init__(api_key)
        self.api = api
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        if api == "alpha_vantage":
            self.alpha_ts = TimeSeries(key=api_key, output_format='pandas')
        if api == "finnhub":
            self.finnhub_client = finnhub.Client(api_key=api_key)
            

    def gather_historical(self, extra_hours=False):
        api_call_count = 0

        # Convert start_date and end_date to datetime objects
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        
        num_months = 2
        if self.api == "alpha_vantage":
            num_months = 1

        rate_cap = 5
        if self.api == "finnhub":
            rate_cap = 60

        for ticker in self.tickers:
            # Initialize an empty DataFrame for concatenation
            all_data = pd.DataFrame()
            current_start_date = start_date
            pulse_start_time = current_start_date.strftime("%Y-%m-%d")


            while current_start_date < end_date:


                # Calculate end date for the current interval
                interval_end_date = current_start_date + relativedelta(months=num_months) # Dependent on API limits

                # Ensure interval_end_date doesn't exceed the final end_date
                if interval_end_date > end_date:
                    interval_end_date = end_date

                # Convert interval dates to strings for API request
                start_date_str = current_start_date.strftime("%Y-%m-%d")
                interval_end_date_str = interval_end_date.strftime("%Y-%m-%d")

                # API call
                if self.api == "polygon":
                    historical = self.polygon_client.list_aggs(ticker=ticker, multiplier=5, timespan="minute", from_=start_date_str, to=interval_end_date_str, limit=50000)
                elif self.api == "alpha_vantage":
                    historical, meta_data = self.alpha_ts.get_intraday(symbol=ticker, interval='5min', outputsize='full')
                elif self.api == "finnhub":
                    historical = self.finnhub_client.stock_candles(symbol=ticker, resolution='5', _from=int(time.mktime(time.strptime(start_date_str, "%Y-%m-%d"))), to=int(time.mktime(time.strptime(interval_end_date_str, "%Y-%m-%d"))))
                else:
                    raise ValueError("Unsupported API")

                
                # Process data
                df = pd.DataFrame(historical)

                # Convert Unix Epoch Standard times to datetime
                if not df.empty:
                    if self.api == "polygon":
                        df['pst'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('America/Los_Angeles')
                    all_data = pd.concat([all_data, df], ignore_index=True)
                
                api_call_count += 1
                
                # Pause every 5 API calls
                if api_call_count % rate_cap == 0:
                    print(f"Pausing for 1 minute. . . [Just pulled data from {pulse_start_time} to {interval_end_date_str} for {ticker}]")
                    pulse_start_time = interval_end_date + timedelta(days=1)
                    time.sleep(60)

                # Move to the next 2-month interval
                current_start_date = interval_end_date + timedelta(days=1)

            # Remove all after-hours stock
            if not all_data.empty:
                if not extra_hours and self.api == "polygon":
                    market_open = pd.Timestamp("09:30:00").time()
                    market_close = pd.Timestamp("16:00:00").time()
                    all_data = all_data[(all_data['pst'].dt.time >= market_open) & (all_data['pst'].dt.time <= market_close)]
            else:
                sys.stderr.write(f"No records found for {ticker}")

            # Save concatenated data for the ticker
            if not all_data.empty:
                all_data.to_csv(f'historical_data/{ticker}.csv', index=False)
                print(f"Data collection for {ticker} complete. {len(all_data)} records found.")
            else:
                self.tickers.remove(ticker)


    # To Do
    def to_sql(self):

        # Define the database path
        db_path = 'historical_data/stockdata.db'
        
        try:
            # Connect to SQLite database
            sqliteConnection = sqlite3.connect(db_path)
            print(f'SQLite connected with {db_path}')

            # List to store DataFrames
            dfs = []

            # Read each CSV file and append DataFrame to list
            for ticker in self.tickers:
                file_path = f'historical_data/{ticker}.csv'
                try:
                    df = pd.read_csv(file_path)
                    df['ticker'] = ticker
                    dfs.append(df)
                except FileNotFoundError:
                    sys.stderr.write(f"File {file_path} not found.\n")
                except pd.errors.EmptyDataError:
                    sys.stderr.write(f"File {file_path} is empty.\n")
                except Exception as e:
                    sys.stderr.write(f"Failed to read {file_path}: {e}\n")

            if not dfs:
                sys.stderr.write("No data files were read.\n")
                return

            # Concatenate all DataFrames into a single DataFrame
            combined_df = pd.concat(dfs, ignore_index=True)

            # Write the combined DataFrame to the SQLite database
            combined_df.to_sql('stockdata', sqliteConnection, if_exists='replace', index=False)

            print("Data successfully written to the database.")
        
        except sqlite3.Error as e:
            sys.stderr.write(f"SQLite error: {e}\n")
        
        finally:

            cursor = sqliteConnection.cursor()

            # Query the number of rows in the stockdata table
            query = "SELECT COUNT(*) FROM stockdata;"
            cursor.execute(query)
            row_count = cursor.fetchone()[0]

            # Print the number of rows
            print(f"Number of rows in the database: {row_count}")

            query = "SELECT * FROM stockdata;"
            cursor.execute(query)
            self.data = pd.read_sql_query(query, sqliteConnection)

            # Close the SQLite connection
            if sqliteConnection:
                sqliteConnection.close()
                print("SQLite connection closed.")



    # To Do
    def impute_data(self):
       pass