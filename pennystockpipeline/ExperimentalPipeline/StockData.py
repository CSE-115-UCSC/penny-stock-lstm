import pandas as pd
import polygon
import finnhub
from alpha_vantage.timeseries import TimeSeries
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import sqlite3
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import yfinance as yf
import numpy as np
import os




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

    def __init__(self, polygon_key, finnhub_key, market, stock_type, rank_by_volume='false'):
        super().__init__(polygon_key)
        self.finnhub_client = finnhub.Client(api_key=finnhub_key)
        self.finnhub_key = finnhub_key
        self.market = market
        self.stock_type = stock_type
        self.tickers = []
        self.rank_by_volume = rank_by_volume

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
                    active=True,  # Ensure active tickers
                    order='desc',  # Descending order (highest volume first)
                    limit=amount  # Limit number retrieved per page
                )
            except:
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

        # Rank by volume if necessary
        if self.rank_by_volume == True:
            self.volume_sort()

    def remove_non_penny_stocks(self):
        # Remove all non-penny stocks from self.tickers
        print(f"Removing all non-penny stocks from {len(self.tickers)} tickers. . .")
        count = 0
        api_calls = 0
        max_api_calls_per_minute = 60
        
        for ticker in self.tickers:
            count += 1
            
            # Check if the API call limit per minute is reached
            if api_calls >= max_api_calls_per_minute:
                print("Reached Finnhub API call limit, sleeping for 60 seconds...")
                time.sleep(60)  # Sleep for 60 seconds
                api_calls = 0  # Reset API call count after sleeping
            
            try:
                last_close = self.finnhub_client.quote(ticker)['c']
                api_calls += 1  # Increment API call count

                if last_close > 5:
                    self.tickers.remove(ticker)

            except Exception as e:
                sys.stderr.write(f"Failed to make get_last_trade API call: {e}\n")
                print(f"Got through {count} stocks")
                break

        print("Done")

    def volume_sort(self):

        # Sort all tickers by volume over last month
        print(f"Sorting {len(self.tickers)} tickers by volume. . .")
        ticker_volumes = []

        for ticker in self.tickers:
            avg_volume = self.get_average_volume(ticker)
            if avg_volume:
                ticker_volumes.append({'Ticker': ticker, 'Average Volume': avg_volume})
        
        # Convert to DataFrame and sort by volume
        volume_df = pd.DataFrame(ticker_volumes)
        ranked_stocks = volume_df.sort_values(by='Average Volume', ascending=False)
        self.tickers = ranked_stocks['Ticker'].tolist()
        print("Done")
        
    
    def get_average_volume(self, ticker, period='3mo'):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval="1d")  # Daily data
            if not hist.empty:
                avg_volume = hist['Volume'].mean()
                return avg_volume
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
        return None



'''Subclass containing list of penny stock tickers using Polygon.io API

Initialization Arguments: 
    - Polygon.io API Key
    - start_date (ex: 2022-01-01)
    - end_data (ex: 2024-01-01)
'''
class Historical(StockData):

    def __init__(self, api, api_key, tickers, start_date, end_date):
        if api_key != None:
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

            if self.api == "finsheet":
                self.finsheet(ticker)
                continue

            # Initialize an empty DataFrame for concatenation
            stock_data = pd.DataFrame()
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
                    stock_data = pd.concat([stock_data, df], ignore_index=True)
                
                api_call_count += 1
                
                # Pause every 5 API calls
                if api_call_count % rate_cap == 0:
                    print(f"Pausing for 1 minute. . . [Just pulled data from {pulse_start_time} to {interval_end_date_str} for {ticker}]")
                    pulse_start_time = interval_end_date + timedelta(days=1)
                    time.sleep(60)

                # Move to the next 2-month interval
                current_start_date = interval_end_date + timedelta(days=1)

            # Remove all after-hours stock
            if not stock_data.empty:
                if not extra_hours and self.api == "polygon":
                    market_open = pd.Timestamp("09:30:00").time()
                    market_close = pd.Timestamp("16:00:00").time()
                    stock_data = stock_data[(stock_data['pst'].dt.time >= market_open) & (stock_data['pst'].dt.time <= market_close)]
            else:
                sys.stderr.write(f"No records found for {ticker}")

            # Save concatenated data for the ticker
            if not stock_data.empty:
                stock_data.to_csv(f'historical_data/{ticker}.csv', index=False)
                print(f"Data collection for {ticker} complete. {len(stock_data)} records found.")
            else:
                self.tickers.remove(ticker)
            
            # Conbine all csv files into single sql database
            # self.to_sql()


    def finsheet(self, ticker, start, end):
        
        excel_path = "historical_data/blankstock.xlsx" 
        output_csv_path = f"historical_data/{ticker}.csv"
        output_csv_name = f"{ticker}.csv"
        columns = ['ftimestamp', 'close', 'open', 'high', 'low', 'volume', 'tp', 'tpxv', 'cumtpxv', 'cumv', 'vwap', 'day', 'time', 'ticker']
        finsheetcall = f'=FS_EquityCandles("{ticker}", "5", "{start}", "{end}", "Period&Close&Open&High&Low&Volume", "-NHCT")'
        # Open the csv
        # Write to A2: =FS_EquityCandles("ticker", "5", "2/01/2021", "3/01/2024", "Period&Close&Open&High&Low&Volume", "-NHCT")
        # Write to entire column N: ticker
        # Write to N1: "ticker"

         # Load the existing workbook
        workbook = load_workbook(excel_path)
        sheet = workbook['Sheet1']

        # Write formula to A2
        formula = f'=FS_EquityCandles("{ticker}", "5", "2/01/2021", "3/01/2024", "Period&Close&Open&High&Low&Volume", "-NHCT")'
        sheet['A2'] = formula

        # Write ticker to column N (starting from N2)
        sheet['N1'] = "ticker"
        sheet['N2'] = ticker

        # Save the changes to the Excel file
        workbook.save(excel_path)

        print(f"Saved {ticker}, sleeping for 5 ...")
        time.sleep(5)

    def to_sql(self):

        # Define the database path
        db_path = 'historical_data/stockdata.db'
        dr = 'historical_data'
        
        try:
            # Connect to SQLite database
            sqliteConnection = sqlite3.connect(db_path)
            print(f'SQLite connected with {db_path}')

            # List to store DataFrames
            dfs = []

            # Read each CSV file and append DataFrame to list
            for filename in os.listdir(dr):
                if filename.endswith('.csv'):
                    ticker = os.path.splitext(filename)[0]
                    file_path = os.path.join(dr, filename)
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
            # print(f"Ended with {len(combined_df['ticker'].unique())} tickers.")

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
    def impute(self, amount, fill, dropout_threshold=0):

        print("Beginning imputation of historical_data/stockdata.db. . .")
        timer_start = time.time()
        # Define the database path
        db_path = 'historical_data/stockdata.db'
        
        try:
            # Connect to SQLite database
            sqliteConnection = sqlite3.connect(db_path)
            print(f'    Debug: SQLite connected with {db_path}')
        
        except sqlite3.Error as e:
            sys.stderr.write(f"SQLite error: {e}\n")
        
        finally:
            cursor = sqliteConnection.cursor()

            # Query to get all data from the stockdata table
            query = "SELECT * FROM stockdata;"
            df = pd.read_sql_query(query, sqliteConnection)
            print(f"Started imputation with {len(df)} rows. Impution method:    amount={amount} | fill={fill} | dropout_threshold={dropout_threshold}")

            # Close the SQLite connection
            if sqliteConnection:
                sqliteConnection.close()
                print('    Debug: SQLite connection closed.')


            imputed_data = pd.DataFrame()

            saved = 0
            discarded = 0
            ratios = []

            for ticker in df['ticker'].unique():
                # Make a copy of df if you are working with a subset
                temp_data = df[df['ticker'] == ticker].copy()

                print(f"Before imputation on {ticker}, it has {len(temp_data)} rows.")
                original_length = len(temp_data)

                # Convert 'timestamp' to datetime
                temp_data['pst'] = pd.to_datetime(temp_data['timestamp'], unit='ms') - pd.Timedelta(hours=8)
                temp_data = temp_data.set_index('pst')

                # Generate a full 5-minute interval timeline for the entire year
                time_index = []

                if amount == "complete":
                    for date in pd.date_range(start=self.start_date, end=self.end_date, freq='D'):
                        start_time = date + pd.Timedelta(hours=9, minutes=30)
                        end_time = date + pd.Timedelta(hours=16)
                        complete_timeline = pd.date_range(start=start_time, end=end_time, freq='5min')
                        time_index.extend(complete_timeline)

                elif amount == "day":
                    unique_dates = temp_data.index.normalize().unique()
                    for date in unique_dates:
                        start_time = date.normalize() + pd.Timedelta(hours=9, minutes=30)
                        end_time = date.normalize() + pd.Timedelta(hours=16)
                        daily_timeline = pd.date_range(start=start_time, end=end_time, freq='5min')
                        time_index.extend(daily_timeline)

                merged_timeline = pd.DataFrame(index=time_index)

                # Filter correct hours (why polygon??)
                temp_data = temp_data.loc[(temp_data.index.time >= start_time.time()) & (temp_data.index.time <= end_time.time())]

                # Combine the time column with existing data
                if len(temp_data) == 0:
                    continue
                temp_imputed = merged_timeline.combine_first(temp_data)

                # Now, apply forward fill to the entire dataset
                if fill == 'forward':
                    temp_imputed.ffill(inplace=True)
                    temp_imputed.bfill(inplace=True)
                elif fill == 'interpolated_linear':
                    temp_imputed.interpolate(method='linear', inplace=True)
                else:
                    raise ValueError("Unsupported interpolation method specified")

                
                ratio = original_length / len(temp_imputed)
                ratios.append(ratio)

                if ratio > dropout_threshold:
                    saved += 1
                    # Append multiple DataFrames row-wise
                    print(f"After imputation on {ticker}, it now has {len(temp_imputed)} rows.")
                    print(f"    Success: ratio of {ratio}")
                    imputed_data = pd.concat([imputed_data, temp_imputed], axis=0, ignore_index=False)
                else:
                    discarded += 1
                    print(f"After imputation on {ticker}, it now has {len(temp_imputed)} rows.")
                    print(f"    Warning: {ticker} does not meet the dropout threshold and will be discarded. It's ratio is {ratio}")

            # Reset the index after all dates are processed
            imputed_data.reset_index(inplace=True)
            imputed_data.rename(columns={'index': 'pst'}, inplace=True)

            try:
                # Sort by ticker, then by time
                imputed_data.sort_values(by=['ticker', 'pst'], inplace=True)
                imputed_data.to_csv('historical_data/stockdata_imputed.csv', index=False)
            
            except Exception:
                sys.stderr.write("Error: Imputation resulted in no usable data.")
                sys.exit()

            # Finish
            self.imputed_data = imputed_data
            timer_end = time.time()
            print(f"\nSaved stocks: {saved}   |   Discarded stocks: {discarded}   |   Average ratio: {sum(ratios) / len(ratios)}")
            print(f"Finished imputation with {len(imputed_data)} total rows in {(timer_end - timer_start):.4f} seconds.")


            # Save the imputed data into a new SQLite database
            output_db_path = 'historical_data/stockdata_imputed.db'
            conn = sqlite3.connect(output_db_path)

            try:
                # Save the DataFrame to a SQL table
                imputed_data.to_sql('stockdata_imputed', conn, if_exists='replace', index=False)
                print(f"    Debug: Imputed data saved successfully to {output_db_path}")
            except sqlite3.Error as e:
                sys.stderr.write(f"    Debug: Error while saving to database: {e}\n")
                sys.exit(1)
            finally:
                # Close the database connection
                if conn:
                    conn.close()
                    print("    Debug: Imputed database connection closed.")
                    
'''Subclass preparing the stock data for model training

Initialization Arguments: 
    - filepath for stock data
'''
class Preprocessor(StockData):
    def __init__(self, filepath):

        sqliteConnection = sqlite3.connect(filepath)
        cursor = sqliteConnection.cursor()
        query = "SELECT * FROM stockdata_imputed;"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = cursor.fetchall()
        table_name = table_names[0][0]
        query = f"SELECT * FROM {table_name};"
        self.data = pd.read_sql_query(query, sqliteConnection)  


