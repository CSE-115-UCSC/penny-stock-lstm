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

import torch # Library for implementing Deep Neural Network 

class PennyStockData():
    
    def __init__(self, database_name_with_path, table_name, impute=True, verbose=0) -> None:
        self.verbose = verbose
        
        self.__load_data(database_name_with_path, table_name, impute)
        self.scaler = MinMaxScaler(feature_range=(0,1))

    def split_dataset(self, split=0.8, to_torch=True):
        x, y = self.xs, self.ys

        #self.train_data, self.test_data = self.psd.o_data[:train_size], self.psd.o_data[train_size:]
        
        train_size = int(len(x) * split)
    
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        #np.array
        print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
        print(f'x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}')
    
        if to_torch:
            x_train = torch.tensor(x_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            x_test = torch.tensor(x_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)

        #print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
        #print(f'x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}')
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        return self

    def create_sequences(self, sequence_length=36, prediction_length=36):
        xs, ys = [], []
        index = 0
        count = 0
        
        normalized_data = self.normalized_data
        self_data = pd.DataFrame(self.data, columns=self.headers)
        
        tickers = self_data['ticker_id']
        dates = self_data['p_date']

        while index < len(normalized_data) - sequence_length - prediction_length + 1:
            # Check if sequence is within a single day
            if dates[index] == dates[index + sequence_length] and tickers[index] == tickers[index + sequence_length]:
                # If day == 2024-05-31, print
                # if dates[index] == "2024-05-31":
                # print("We got a sequence from", dates[index], "to", dates[index + sequence_length], "sequence-length is", (index + sequence_length) -index, tickers[index], tickers[index + sequence_length])
                xs.append(normalized_data[index:index + sequence_length])
                ys.append(normalized_data[index + sequence_length:index + sequence_length + prediction_length])
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
            print(f'[INFO][PennyStockData]: Performing global normalization on {columns_to_normalize} using MixMaxScaler')
        
        normalized_data[columns_to_normalize] = self.scaler.fit_transform(normalized_data[columns_to_normalize])
        
        #normalized_data = normalized_data.drop(columns=['ticker_id'])
        #normalized_data.reset_index(drop=True, inplace=True)
        
        self.normalized_data = normalized_data[columns_to_normalize].values.tolist()
        self.normalized_headers = columns_to_normalize
        
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
    
        query = "SELECT ticker_id, ticker, p_date, p_time, volume_weighted_average, open, close, high, low, time/1000 as time, volume, number_of_trades FROM " + table + " WHERE ticker_id<>30 ORDER BY ticker_id, p_date, p_time"
        cursor.execute(query)
        
        data = [list(i) for i in cursor.fetchall()]
        headers = list(map(lambda x: x[0], cursor.description))
        
        if impute:
            data, headers = self._get_imputed(data, headers)

        self.data = data
        self.headers = headers

        # get max next date from database
        next_max_date = self.__get_next_max_date(table, cursor)
        print(next_max_date)
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
