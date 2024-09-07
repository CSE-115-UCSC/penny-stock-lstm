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

import matplotlib.pyplot as plt # Visualization 
import matplotlib.dates as mdates # Formatting dates
import seaborn as sns # Visualization

from sklearn.preprocessing import MinMaxScaler

import torch # Library for implementing Deep Neural Network 

class PennyStockData():
    
    def __init__(self, database_name_with_path, table_name, impute=True, verbose=0) -> None:
        #torch.manual_seed(1)
        self.verbose = verbose
        
        self.__load_data(database_name_with_path, table_name, impute)
        self.scaler = MinMaxScaler(feature_range=(0,1))

    def plot_data(self):
        plt.rcParams['figure.figsize'] = [14, 4] 
        
        data_df = pd.DataFrame(self.data, columns = self.headers)
        #data_dfn = pd.DataFrame(self.normalized_data, columns = self.normalized_headers)
        plt.plot(data_df['p_date'], data_df['volume_weighted_average'], label = "volume_weighted_average", color = "b")
        
        return self

    #def split_dataset(self, split=0.8, to_torch=True):
    #    x, y = self.xs, self.ys

        #self.train_data, self.test_data = self.psd.o_data[:train_size], self.psd.o_data[train_size:]
        
    #    train_size = int(len(x) * split)
    
    #    x_train, x_test = x[:train_size], x[train_size:]
    #    y_train, y_test = y[:train_size], y[train_size:]
        #np.array
    #    print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
    #    print(f'x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}')
    
    #    if to_torch:
    #        x_train = torch.tensor(x_train, dtype=torch.float32)
    #        y_train = torch.tensor(y_train, dtype=torch.float32)
    #        x_test = torch.tensor(x_test, dtype=torch.float32)
    #        y_test = torch.tensor(y_test, dtype=torch.float32)

        #print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
        #print(f'x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}')
        
    #    self.x_train = x_train
    #    self.x_test = x_test
    #    self.y_train = y_train
    #    self.y_test = y_test
        
    #    return self

    def split_and_create_sequences(self, sequence_length=36, prediction_length=36, train_test_split_at=29, to_torch=True):
        x_train, y_train, x_test, y_test = [], [], [], []
        
        normalized_data = self.normalized_data
        normalized_headers = self.normalized_headers
        
        self_data = pd.DataFrame(self.data, columns=self.headers)

        train_set = self_data[self_data['ticker_id'] < train_test_split_at]
        test_set = self_data[self_data['ticker_id'] >= train_test_split_at]

        train_set.reset_index(drop=True, inplace=True)
        test_set.reset_index(drop=True, inplace=True)

        #print(f'{len(train_set)}/{len(test_set)}')

        #return self

        train_tickers = train_set['ticker_id']
        test_tickers = test_set['ticker_id']
        
        train_dates = train_set['p_date']
        test_dates = test_set['p_date']

        train_index = 0
        train_count = 0

        while train_index < (len(train_tickers) - sequence_length - 1):
            if train_dates[train_index] == train_dates[train_index+sequence_length+1] and train_tickers[train_index] == train_tickers[train_index+sequence_length+1]:
                x_train.append(normalized_data[train_index:train_index+sequence_length])
                y_train.append(normalized_data[train_index+sequence_length+1])  # Predicting the value right after the sequence

                train_index += 1
                train_count += 1
            else:
                newindex = train_index
                while train_dates[newindex] == train_dates[newindex + 1]:
                    newindex += 1
                newindex += 1
                train_index = newindex

        test_index = 0
        test_count = 0

        while test_index < (len(test_tickers) - prediction_length - 1):
            if test_dates[test_index] == test_dates[test_index+prediction_length+1] and test_tickers[test_index] == test_tickers[test_index+prediction_length+1]:
                x_test.append(normalized_data[test_index:test_index+prediction_length])
                y_test.append(normalized_data[test_index+prediction_length+1])  # Predicting the value right after the sequence

                test_index += 1
                test_count += 1
            else:
                newindex = test_index
                while test_dates[newindex] == test_dates[newindex + 1]:
                    newindex += 1
                newindex += 1
                test_index = newindex
        
        #while train_index < len(train_tickers) - sequence_length + 1:
        #while train_index < len(train_tickers) - sequence_length - 1:
            #print(f'train_index: {train_index}')
            # Check if sequence is within a single day
        #    if train_dates[train_index] == train_dates[train_index+sequence_length] and train_tickers[train_index] == train_tickers[train_index+sequence_length]:
        #        x_train.append(normalized_data[train_index:train_index+sequence_length])
        #        y_train.append(normalized_data[train_index+sequence_length])

        #        train_index += sequence_length
        #        train_count += 1
        #    else:  # Move index to the start of the next day
        #        newindex = train_index
        #        while train_dates[newindex] == train_dates[newindex + 1]:
        #            newindex += 1
        #        newindex += 1
        #        train_index = newindex

        #test_index = 0
        #test_count = 0

        #while test_index < len(test_tickers) - prediction_length - 1:
            #print(f'test_index: {test_index}')
            # Check if sequence is within a single day
        #    if test_dates[test_index] == test_dates[test_index+prediction_length] and test_tickers[test_index] == test_tickers[test_index+prediction_length]:
        #        x_test.append(normalized_data[test_index:test_index+prediction_length])
        #        y_test.append(normalized_data[test_index+prediction_length])

        #        test_index += prediction_length
        #        test_count += 1
        #    else:  # Move index to the start of the next day
        #        newindex = test_index
        #        while test_dates[newindex] == test_dates[newindex + 1]:
        #            newindex += 1
        #        newindex += 1
        #        test_index = newindex

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        #x_train = np.reshape(x_train, (-1, 1))
        #y_train = np.reshape(y_train, (-1, 1))
        #x_test = np.reshape(x_test, (-1, 1))
        #y_test = np.reshape(y_test, (-1, 1))

        if to_torch:
            x_train = torch.tensor(x_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            x_test = torch.tensor(x_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        print(f'train_count: {train_count} test_count: {test_count}')
        print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
        print(f'x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}')

        return self
    
    def normalize(self, columns_to_normalize=[]):
        if (len(columns_to_normalize) == 0):
            print(f'[INFO][PennyStockData]: No columns were supplied to be normalized {columns_to_normalize}. Provide columns as a list in columns_to_normalize')
            return self

        if (self.verbose == 2):
            print(f'[INFO][PennyStockData]: Performing ticker-wise normalization on {columns_to_normalize}')

        normalized_data = pd.DataFrame()
        # First we normalize by each ticker as tickerwise, the wva differs a lot
        dfx = pd.DataFrame(self.data, columns=self.headers)
        
        data_by_ticker = {}
        for ticker in dfx['ticker_id'].unique():
            data_by_ticker[ticker] = dfx[dfx['ticker_id'] == ticker].copy()
            for ctn in columns_to_normalize:
                if ctn in dfx.columns:
                    data_by_ticker[ticker][ctn] = (data_by_ticker[ticker][ctn] / data_by_ticker[ticker][ctn].max()) ## doing inplace
    
        for ticker in data_by_ticker:
            # create a temporary DataFrame to hold the current data
            temp_df = pd.DataFrame(data_by_ticker[ticker].values, columns=data_by_ticker[ticker].keys())
            normalized_data = pd.concat([normalized_data, temp_df], axis=0, ignore_index=True)
    
        # optionally, you can reset the index if needed
        normalized_data.reset_index(drop=True, inplace=True)

        if (self.verbose == 2):
            print(f'[INFO][PennyStockData]: Performing global normalization on {columns_to_normalize} using MinMaxScaler')
        
        normalized_data[columns_to_normalize] = self.scaler.fit_transform(normalized_data[columns_to_normalize])
        
        #normalized_data = normalized_data.drop(columns=['ticker_id'])
        #normalized_data.reset_index(drop=True, inplace=True)
        
        #self.normalized_data = normalized_data[columns_to_normalize].values.tolist()
        #self.normalized_headers = columns_to_normalize

        #self.normalized_columns = columns_to_normalize
        
        self.normalized_data = normalized_data[columns_to_normalize].values.tolist()
        self.normalized_headers = columns_to_normalize

        #print(self.normalized_headers)
        
        return self

    def get_columns(self, columns_as_list=[]):
        if (len(columns_as_list) == 0):
            return self
        
        df = pd.DataFrame(self.data, columns=self.headers)
        
        self.data = df[columns_as_list].values.tolist()
        self.headers = columns_as_list

        return self

    def _get_imputed(self, data, headers) -> (list, list):
        no_of_records = len(data)
        total_imputed = 0
        
        # First row
        time_diff = 0
        data[0].append(time_diff)
        
        imputed_data = list()
        imputed_data.append(data[0])
        
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
                    copied_record_from_previous_row[9] = (int)(copied_record_from_previous_row[9]) + (300 * imputed_cursor) ## add 5 mins
                    copied_record_from_previous_row[3] = strftime("%H:%M", gmtime(((int)(copied_record_from_previous_row[9]))))
                    copied_record_from_previous_row.append(5.)
                    
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

        print(f'[DEBUG][PennyStockData]: Imputed len(data): {len(imputed_data)}')
        
        return imputed_data, headers

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
    def __load_data(self, path, table, impute):
        assert self.__file_exists(path)
        data = None
        
        sqliteConnection = sqlite3.connect(path)
        cursor = sqliteConnection.cursor()
    
        query = "SELECT ticker_id, ticker, p_date, p_time, volume_weighted_average, open, close, high, low, time/1000 as time, volume, number_of_trades FROM " + table + " WHERE ticker_id <> 30 ORDER BY ticker_id, p_date, p_time"
        cursor.execute(query)
        
        data = [list(i) for i in cursor.fetchall()]
        headers = list(map(lambda x: x[0], cursor.description))
        
        if impute:
            data, headers = self._get_imputed(data, headers)

        self.data = data
        self.headers = headers

        # get max next date from database
        next_max_date = self.__get_next_max_date(table, cursor)
        print(f'next_max_date: {next_max_date}')
        self.next_max_date = next_max_date
        
        return self

    def __get_next_max_date(self, table, cursor) -> str:
        next_max_date = ""
        query = "SELECT max(date(p_date, '+1 day')) as next_max_date FROM " + table + " WHERE 1"
        cursor.execute(query)
        
        rs = cursor.fetchone()
        if (rs is None):
            raise Exception("Max Date could not be retrieved from DB")
        else:
            next_max_date = rs[0]

        return next_max_date
    
    # Private valid file checker
    def __file_exists(self, path) -> bool:
        if os.path.isfile(path):
            return True
        else:
            print("[INFO][PennyStockData]:", path, "not found")
            return False
