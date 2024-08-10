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
from sklearn.preprocessing import MinMaxScaler

from time import time, strftime, gmtime

class PennyStockData:
    
    def __init__(self, database_name_with_path, table_name, impute=True, format='csv', verbose=0):
        ## Extract
        #assert (database_name_with_path and database_name_with_path.strip()) == ""
        #assert (table_name and table_name.strip()) == ""

        self.verbose = verbose
        self.path = self.file_exists(database_name_with_path)
        
        self.impute = impute
        self.format = format
        
        self.sqliteConnection, self.cursor = self.connect_db()
        self.table_name = table_name
        
        self.data, self.headers, self.size = self.load_data()
        self.tickers = self.get_tickers()
        self.data = self.get_imputed()
        

        
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
        
        return imputed_data

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
    
    def __len__(self):
        return len(self.data)

    def get_tickers(self):
        tickers = []
        try:
          query = "SELECT id, ticker FROM tickers WHERE 1 ORDER BY id"
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
          query = "SELECT ticker_id, ticker, p_date, p_time, volume_weighted_average, open, close, high, low, time/1000 as time, volume, number_of_trades FROM " + self.table_name + " WHERE 1 ORDER BY ticker_id, p_date, p_time"
          self.cursor.execute(query)
          if self.verbose >= 1:
            print(f'[INFO][PennyStockData]: SQlite executed query {query}')
          #data = pd.read_sql_query(query, self.sqliteConnection)
          #headers = data.columns
          #size = len(data)
          
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

