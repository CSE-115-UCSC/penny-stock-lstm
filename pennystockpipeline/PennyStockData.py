## #!/usr/bin/env python

## PennyStockData
## This pipeline performs the following operations on data:
##  - Data Gathering
##  - Data Cleaning and Transformation
##  - (Later, mat be) Exploratory Data analysis (EDA)

## ToDo: Allow specifying features list to be selected from the database table or should we use Torch Filter

## Please refer to the PennyStockPipeline.md for details of this pipeline

# import torchdata.datapipes as dp
import csv, os, sys
import sqlite3
import numpy as np
import pandas as pd
from time import time, strftime, gmtime

from sklearn.preprocessing import MinMaxScaler

class PennyStockData():
    
    def __init__(self, database_name_with_path, table_name, impute=True, verbose=0):
        self.verbose = verbose
        self.path = self.file_exists(database_name_with_path)
        
        self.impute = impute
        
        self.sqliteConnection, self.cursor = self.connect_db()
        self.table_name = table_name
        
        self.data, self.headers, self.size = self.load_data()
        self.tickers = self.get_all_tickers()
        self.selected_tickers = self.get_selected_tickers()
        
        self.data = self.get_imputed()

        self.xs = None
        self.ys = None

    def create_sequences(self, sequence_length=78, prediction_length=78):
        xs, ys = [], []
        index = 0
        count = 0
        
        data_df = pd.DataFrame(self.normalized_data, columns=self.normalized_headers)
        data = data_df[self.columns_to_normalize].values.tolist()
        #data = data.values.tolist()
        
        tickers = self.ds_tickers
        dates = self.ds_dates
        
        while index < len(data) - sequence_length - prediction_length + 1:
            # Check if sequence is within a single day
            if dates[index] == dates[index + sequence_length] and tickers[index] == tickers[index + sequence_length]:
                # If day == 2024-05-31, print
                # if dates[index] == "2024-05-31":
                # print("We got a sequence from", dates[index], "to", dates[index + sequence_length], "sequence-length is", (index + sequence_length) -index, tickers[index], tickers[index + sequence_length])
                xs.append(data[index:index + sequence_length])
                ys.append(data[index + sequence_length:index + sequence_length + prediction_length])
                index += sequence_length
                count += 1
            else:  # Move index to the start of the next day
                newindex = index
                while dates[newindex] == dates[newindex + 1]:
                    newindex += 1
                newindex += 1
                index = newindex
        #print("Valid days:", count)

        self.xs = np.array(xs)
        self.ys = np.array(ys)

        #self.data = data

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        return self
    
    def normalize(self, columns_to_normalize=[]):
        print("Normalizing...")
        if (len(columns_to_normalize) == 0):
            print(f'[INFO][PennyStockData]: No columns were supplied to be normalized {columns_to_normalize}. Provide columns as a list in columns_to_normalize')
            print("Returning early")
            return self

        if (self.verbose == 2):
            print(f'[INFO][PennyStockData]: Performing ticker-wise normalization on {columns_to_normalize}')

        ## Storing original data and headers
        #self.o_data = self.data
        #self.o_headers = self.headers
        
        # normalized_data = pd.DataFrame()
        # First we normalize by each ticker as tickerwise, the wva differs a lot
            
        # Create a scaler for every stock
        scalers = {}
        data_by_ticker = {}
        normalized_data = pd.DataFrame()
        dfx = pd.DataFrame(self.data, columns=self.headers)
        for ticker in self.get_selected_tickers():
            
            # Separate data by ticker and then by column
            data_by_ticker[ticker] = dfx[dfx['ticker_id'] == ticker].copy() # Shape is (40840, 4)
            data_to_normalize = data_by_ticker[ticker][columns_to_normalize]


            
            # Scale the selected columns individually based on ticker
            scalers[ticker] = MinMaxScaler(feature_range=(0,1)) 
            normalized_column_data = scalers[ticker].fit_transform(data_to_normalize)
            
            # Reinsert the normalized data into the table
            data_by_ticker[ticker][columns_to_normalize] = normalized_column_data
            temp_df = pd.DataFrame(data_by_ticker[ticker].values.tolist(), columns=data_by_ticker[ticker].keys().tolist())
            normalized_data = pd.concat([normalized_data, temp_df], axis=0, ignore_index=True)

        # optionally, you can reset the index if needed
        normalized_data.reset_index(drop=True, inplace=True)
        

        if (self.verbose == 2):
            print(f'[INFO][PennyStockData]: Performing global normalization on {columns_to_normalize} using MixMaxScaler')
        

        # normalized_data[columns_to_normalize] = scaler[].fit_transform(normalized_data[columns_to_normalize])

        self.ds_tickers = normalized_data['ticker_id'].values.tolist()
        self.ds_dates = normalized_data['p_date'].values.tolist()
        self.ds_times = normalized_data['p_time'].values.tolist()

        normalized_data = normalized_data.drop(columns=['ticker_id'])
        normalized_data.reset_index(drop=True, inplace=True)
        
        self.normalized_data = normalized_data.values.tolist()
        self.normalized_headers = normalized_data.columns.tolist()

        self.columns_to_normalize = columns_to_normalize

        self.scaler = scalers # Change -> Array
        
        return self

    def get_headers(self):
        return self.headers

    def get_columns(self, columns_as_list=[]):
        if (len(columns_as_list) == 0):
            return self.data
        
        df = pd.DataFrame(self.data, columns=self.headers)
        to_df = df[columns_as_list]
        
        self.data = to_df.values.tolist()
        
        to_headers = to_df.columns
        self.headers = to_headers.tolist()
        
        return self

    def get_selected_tickers(self):
        df = pd.DataFrame(self.data, columns=self.headers)

        return df["ticker_id"].unique().tolist()
        
    def get_imputed(self):
        if not(self.impute):
            return self.data

        data = self.data
        no_of_records = len(data)
        total_imputed = 0
        # First row
        time_diff = 0
        data[0].append(time_diff)
        imputed_data = list()
        imputed_data.append(data[0])
        headers = self.headers
        
        for row_index in range(no_of_records-1):
            if (data[row_index][0] == data[row_index+1][0] and data[row_index][2] == data[row_index+1][2]):
                time_diff = (int)(((int)(data[row_index+1][9]) - (int)(data[row_index][9]))/(60))  # in mins
            else:
                time_diff = 0
            
            # IMPUTE
            if (time_diff > 5):
                rows_to_impute = (int)((time_diff - 5) / 5)
                imputed_cursor = 1
        
                final_row = data[row_index+1].copy()
                final_row.append(5.)
                
                for o in range(rows_to_impute):
                    ## we need to insert this copied_record_from_previous_row after data[row_index]
                    copied_record_from_previous_row = None
                    copied_record_from_previous_row = data[row_index].copy()
                    copied_record_from_previous_row.append(5.)
                    #copied_record_from_previous_row[10] = 0
                    #copied_record_from_previous_row[11] = 0
                    copied_record_from_previous_row[9] = (int)(copied_record_from_previous_row[9]) + (300 * imputed_cursor) ## add 5 mins
                    copied_record_from_previous_row[3] = strftime("%H:%M", gmtime(((int)(copied_record_from_previous_row[9]))))
                    imputed_data.append(copied_record_from_previous_row)
                    
                    copied_record_from_previous_row = None
                    imputed_cursor = imputed_cursor + 1
                    
                imputed_data.append(final_row)
                total_imputed = total_imputed + rows_to_impute  
            else:
                copied_record_from_previous_row = data[row_index+1].copy()
                copied_record_from_previous_row.append(time_diff)
                imputed_data.append(copied_record_from_previous_row)
            
        headers.append('time-diff')
        self.headers = headers

        print(f'[DEBUG][PennyStockData]: Imputed len(data): {len(imputed_data)}')
        
        return imputed_data

    def __len__(self):
        return len(self.data)

    def get_all_tickers(self):
        tickers = []
        try:
          query = "SELECT id, ticker FROM tickers WHERE id<>30 ORDER BY id"
          self.cursor.execute(query)
          if self.verbose >= 1:
            print(f'[INFO][PennyStockData]: SQlite executed query {query}')
          
          tickers = [list(i) for i in self.cursor.fetchall()]
        except:
          sys.stderr.write("[ERROR][PennyStockData]: Failed to execute query")
        return tickers
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    # Connect to SQlite database
    def connect_db(self):
        cursor = None
        try:
          sqliteConnection = sqlite3.connect(self.path)
          cursor = sqliteConnection.cursor()
          if self.verbose >= 1:
            print(f'[INFO][PennyStockData]: SQlite connected with {self.path}')          
        except:
          raise Error('[ERROR][PennyStockData]: Failed to connect with {}.'.format(self.path))
        return sqliteConnection, cursor

    def load_data(self):
        data = []
        headers = []
        size = 0
        # Execute query
        try:
          query = "SELECT ticker_id, ticker, p_date, p_time, volume_weighted_average, open, close, high, low, time/1000 as time, volume, number_of_trades FROM " + self.table_name + " WHERE ticker_id<>30 ORDER BY ticker_id, p_date, p_time"
          self.cursor.execute(query)
          if self.verbose >= 1:
            print(f'[INFO][PennyStockData]: SQlite executed query {query}')
          
          data = [list(i) for i in self.cursor.fetchall()]
          headers = list(map(lambda x: x[0], self.cursor.description))
          size = len(data)

          if self.verbose == 2:
            print(f'[DEBUG][PennyStockData]: headers: {headers}')
            print(f'[DEBUG][PennyStockData]: len(data): {size}')
        except:
          sys.stderr.write("[ERROR][PennyStockData]: Failed to execute query")

        return data, headers, size

    # Private valid file checker
    def file_exists(self, path):
        if os.path.isfile(path):
            if self.verbose >= 1:
                print("[INFO][PennyStockData]:", path, "exists")
            return path
        else:
            raise FileNotFoundError('[ERROR][PennyStockData]: {} is not found.'.format(path))

'''
    def as_numpy_sequence(self):
        if (self.format == 'csv'):
            return self.data
        elif (self.format == 'numpy_sequence'):
            tickers = np.array(self.tickers)
            t_data = np.transpose(self.data)
            data = np.array(self.data)
            tickers_data = {}

            ## The idea is to create an array such that dataset has structure [ticker_id][p_date][(col0, col1, col2, ..., coln),(col0, col1, col2, ..., coln)]
            ## here (col0, col1, col2, ..., coln) is the sequence data as a list of tuples from each row having 5 mins interval data 
            ## (refer imputing for missing data)

            for ticker in tickers:
                ticker_indices = np.where(t_data[0] == (str)(ticker[0]))
                ticker_data = data[ticker_indices]
                ticker_dates = np.unique(ticker_data[:,2])
            
                ticker_dates_data = {}
                
                for ticker_date in ticker_dates:
                    ticker_date_indices = np.where(ticker_data[:,2] == (str)(ticker_date))
                    ticker_date_data = data[ticker_date_indices]
                    
                    ticker_dates_data[ticker_date] = ticker_date_data
                tickers_data[ticker[0]] = ticker_dates_data

            dataset = tickers_data
            
            if self.verbose >= 1:
                print(f'[INFO][PennyStockData]: Records have been numpied successfully on the variable dataset')
            return dataset
            
        else:
            raise ValueError('[ERROR][PennyStockData]: {} should either be csv or numpy_sequence.'.format(self.format))
'''