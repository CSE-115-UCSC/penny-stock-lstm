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

from time import time, strftime

class PennyStockData:

    __dataset_types = ['csv', 'numpy_sequence']
    
    def __init__(self, database_name_with_path, table_name, return_data_as='csv', verbose=0):
        ## Extract
        #assert (database_name_with_path and database_name_with_path.strip()) == ""
        #assert (table_name and table_name.strip()) == ""

        self.verbose = verbose
        self.data_format = return_data_as
        self.path = self.file_exists(database_name_with_path)
        
        self.sqliteConnection, self.cursor = self.connect_db()
        self.table_name = table_name
        self.data, self.headers, self.size = self.load_data()

        self.tickers = self.get_tickers()

        self.dataset = self.as_numpy(impute=True, row_data_threshold=25)

        #self.data = self.process_data()
        #if (self.data_format == 'numpy_sequence'):
        #   self.data = self.get_data_as_numpy_sequence() 

        
        #self.ticker_data = self.get_data_by_tickers()

        # [np.where(np.transpose(psd.data)[0] == t) for t in tickers] for j in range(psd.size)

        ## Transform
        ## Use Torch Transforms to clean and formatting the data
        ## torchvision.transforms

        ## Load
        ## Use Torch Datasets
        ## To Do: Get input for load Filter

    #def get_data_as_numpy_sequence(self):
    #    data = []

        
    #    return data

    def as_numpy(self, impute=True, row_data_threshold=25):
        # let's create the dataset
        all_historical_dataset = list()
        ticker_dates_list = list()
        ticker_date_data_list = list()
        
        dataset = list()
        #uncompressed_dataset = list()

        tickers_list = self.tickers
        ticker_data = list()
        for ticker in tickers_list:
            #print(ticker)
            ticker_query = "select distinct(p_date) as ticker_date from " + self.table_name + " where ticker_id=" + (str)(ticker[0]) + " order by ticker_date;"
            self.cursor.execute(ticker_query)
            ticker_dates_list = [list(j) for j in self.cursor.fetchall()]
            date_data = list()
            for ticker_date in ticker_dates_list:
                #print(f'ticker_date {ticker_date[0]}')
                ticker_date_query = "SELECT p_time, volume, volume_weighted_average, open, close, high, low, time, number_of_trades FROM " + self.table_name + " WHERE ticker_id=" + (str)(ticker[0]) + " and p_date = '" + ticker_date[0] + "' ORDER BY p_time;"
                #ticker_date_query = "select strftime('%H:%M', time/1000, 'unixepoch') as ticker_time, volume, volume_weighted_average, time from all_historical where ticker='" + ticker + "' and strftime('%Y-%m-%d', time/1000, 'unixepoch') = '" + ticker_date + "' order by ticker_time;" 
                self.cursor.execute(ticker_date_query)
                ticker_date_data_list = [list(k) for k in self.cursor.fetchall()]
                row_data = list()
                for ticker_date_data_row in ticker_date_data_list:
                    # avoid adding data that has zero records or less than 60% from a day (25.2/42 records, starting at 4:30 ending at 8:00)
                    if (len(ticker_date_data_list) > row_data_threshold):
                        ticker_date_data_row_0 = ticker_date_data_row[0].replace(":","")
                        row_data.append(((int)(ticker_date_data_row_0), (int)(ticker_date_data_row[1]), (float)(ticker_date_data_row[2]), (float)(ticker_date_data_row[2]), (float)(ticker_date_data_row[3]), (float)(ticker_date_data_row[4]), (float)(ticker_date_data_row[5]), (float)(ticker_date_data_row[6]), (int)(ticker_date_data_row[7]), (int)(ticker_date_data_row[8])))
                    ## if impute
                    else:
                        if self.verbose == 2:
                            print(f'[DEBUG][PennyStockData]: dropping data ({ticker[0]}, {ticker_date[0]}) rows={len(ticker_date_data_list)} is < row_data_threshold: {row_data_threshold}')
                        break
                
                #if (len(row_data) > row_data_threshold):
                date_data.append((ticker[0], ticker_date[0], row_data))

                    #with open(csv_filename, 'a', newline='') as file:
                    #    csvwriter = csv.writer(file)
                    #    csvwriter.writerow((ticker[0], ticker_date[0], row_data, row_label))
                    #file.close()
            ticker_data.append((date_data))
        #print(ticker_data)
        dataset = ticker_data

        if self.verbose >= 1:
            print(f'[INFO][PennyStockData]: Records have been numpied successfully on the variable dataset')
        return dataset

    def __len__(self):
        return len(self.data)

    def impute(self):
        return self.data

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
 
    #def get_data_by_tickers(self):
    #    tickers = self.get_tickers()
    #    ticker_data = {}
    #    transposed_data = np.transpose(self.data)
    #    for ticker in tickers:
    #        ticker_data[ticker] = transposed_data[np.where(transposed_data[1] == ticker)]

    #    ticker_data = np.transpose(ticker_data)
        
    #    return ticker_data
        
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
          query = "SELECT ROWID, ticker_id, ticker, p_date, p_time, volume, volume_weighted_average, open, close, high, low, time, number_of_trades FROM " + self.table_name + " WHERE 1 ORDER BY ticker_id, p_date, p_time"
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
            print(f'[DEBUG][PennyStockData]: len(data): > {size}')
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
                if (impute):
                    first_hm_price = 0.0 # actually it's 0
                    previous_hm = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan) # actually it's 0
                    row_data = list()
                    f = 0
                    is_first_hm_price_set = False
                    for ticker_date_data_row in ticker_date_data_list:
                        #print(ticker_date_data_row)
                        # Let's skip the 0 hours data
                        f = f + 1
                        if ((ticker_date_data_row[0]) == '00:00'):
                            continue
                        
                        # if (first_hm_price == 0. and first_hm_price < (float)(ticker_date_data_list[f][2])):
                        if (not(is_first_hm_price_set)):
                            first_hm_price = (float)(ticker_date_data_list[f][2])
                            is_first_hm_price_set = True
                        
                        # If previous time is more than 5 seconds but not 0000 hours, we loop to impute values by previous rows
                        if (previous_hm != (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)):
                            time_diff = (int)((ticker_date_data_row[7] - previous_hm[7])/(1000 * 60 * 5))
                            if time_diff > 1:
                                for td in range (time_diff-1):
                                    previous_hm_0_plus_5 = ((int)(previous_hm[0])+5)
                                    if (previous_hm_0_plus_5%100==60):
                                        previous_hm_0_plus_5 = (int)(previous_hm[0])+45 #(+100-55)
                                    row_data.append(((previous_hm_0_plus_5), (int)(previous_hm[1]), (float)(previous_hm[2]), (float)(previous_hm[3]), (float)(previous_hm[4]), (float)(previous_hm[5]), (float)(previous_hm[6]), (int)(previous_hm[7]), (int)(previous_hm[8])))
                                    #row_label.append((label))  ### Since, this is just copying previous rows, retaining price raise label
                                    previous_hm = ((previous_hm_0_plus_5), (int)(previous_hm[1]), (float)(previous_hm[2]), (float)(previous_hm[3]), (float)(previous_hm[4]), (float)(previous_hm[5]), (float)(previous_hm[6]), (int)(previous_hm[7]), (int)(previous_hm[8]))
                                    # time_diff = time_diff - 1
                                    #uncompressed_dataset.append((ticker[0], ticker_date[0], (previous_hm_0_plus_5), (int)(previous_hm[1]), (float)(previous_hm[2]), (float)(previous_hm[3]), (float)(previous_hm[4]), (float)(previous_hm[5]), (float)(previous_hm[6]), (int)(previous_hm[7]), (int)(previous_hm[8])))
            
                            # parallely, we label the row as buy if there's a 30% raise in price, default=no_buy
                            
                            #price_change_percentage = float((float(ticker_date_data_row[2] - first_hm_price) * 100)/float(first_hm_price))
                            #if price_change_percentage >= gain_threshold:  # price gain/raise
                            #    label = "buy"
            
                        ticker_date_data_row_0 = ticker_date_data_row[0].replace(":","")
                        row_data.append(((int)(ticker_date_data_row_0), (int)(ticker_date_data_row[1]), (float)(ticker_date_data_row[2]), (float)(ticker_date_data_row[2]), (float)(ticker_date_data_row[3]), (float)(ticker_date_data_row[4]), (float)(ticker_date_data_row[5]), (float)(ticker_date_data_row[6]), (int)(ticker_date_data_row[7]), (int)(ticker_date_data_row[8])))
                        #row_label.append((label))
                        previous_hm = ((int)(ticker_date_data_row_0), (int)(ticker_date_data_row[1]), (float)(ticker_date_data_row[2]), (float)(ticker_date_data_row[2]), (float)(ticker_date_data_row[3]), (float)(ticker_date_data_row[4]), (float)(ticker_date_data_row[5]), (float)(ticker_date_data_row[6]), (int)(ticker_date_data_row[7]), (int)(ticker_date_data_row[8]))
                        #uncompressed_dataset.append((ticker[0], ticker_date[0], (int)(ticker_date_data_row_0), (float)(ticker_date_data_row[1]), (float)(ticker_date_data_row[2])))
                    #print(row_data)
                    #exit(1)
'''
