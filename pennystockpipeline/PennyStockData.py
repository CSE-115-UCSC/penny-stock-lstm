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
    
    def __init__(self, database_name_with_path, table_name, drop_discrete_columns=True, impute=True, row_data_threshold=12, format='csv', verbose=0):
        ## Extract
        #assert (database_name_with_path and database_name_with_path.strip()) == ""
        #assert (table_name and table_name.strip()) == ""

        self.verbose = verbose
        self.path = self.file_exists(database_name_with_path)
        self.drop_discrete_columns = drop_discrete_columns
        self.impute = impute
        self.row_data_threshold = row_data_threshold
        self.format = format
        
        self.sqliteConnection, self.cursor = self.connect_db()
        self.table_name = table_name
        
        self.data, self.headers, self.size = self.load_data()

        self.tickers = self.get_tickers()

        self.data = self.get_imputed()

        self.size = len(self.data)
        
        self.numpy_sequence = self.as_numpy_sequence()

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

    def get_imputed(self):
        if not(self.impute):
            return self.data
        
        data = self.data

        #ticker_id_list = list()
        no_of_records = len(data)
        row_length = len(data[0])
        t_data = np.transpose(data)
        new_column = np.zeros((no_of_records, 1))
        # First row
        time_diff = 0
        data[0].append(time_diff)
        
        for row_index in range(no_of_records-1):
            
            if (data[row_index][1] == data[row_index+1][1] and data[row_index][3] == data[row_index+1][3]):
                time_diff = ((int)(data[row_index+1][10]) - (int)(data[row_index][10]))/(1000 * 60)  # in mins
            else:
                time_diff = 0
            data[row_index+1].append(time_diff)
        self.headers.append('time-diff')

        # ToDo:
        ## for each row, loop and insert data
        ## tip: use the previous loop

        ## consider force refusing impute when discrete fields (i.e. volume, number_of_sales) are in the column_list
        ## i.e. if drop_discrete_columns == False, do not allow imputing (as we only add missing 5th minute data to rows)
        
        return data

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
                ticker_indices = np.where(t_data[1] == (str)(ticker[0]))
                ticker_data = data[ticker_indices]
                ticker_dates = np.unique(ticker_data[:,3])
            
                ticker_dates_data = {}
                
                for ticker_date in ticker_dates:
                    ticker_date_indices = np.where(ticker_data[:,3] == (str)(ticker_date))
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

    #def impute(self):
    #    return self.data

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
          discrete_columns = ""
          if (not(self.drop_discrete_columns)):
            discrete_columns = ", volume, number_of_trades"
          query = "SELECT ROWID, ticker_id, ticker, p_date, p_time, volume_weighted_average, open, close, high, low, time" + discrete_columns + " FROM " + self.table_name + " WHERE 1 ORDER BY ticker_id, p_date, p_time"
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


############################### numpy_sequence - old_code

# let's create the dataset
            all_historical_dataset = list()
            ticker_dates_list = list()
            ticker_date_data_list = list()
            
            dataset = None
            #uncompressed_dataset = list()
    
            tickers_list = self.tickers
            ticker_data = None
            for ticker in tickers_list:
                #print(ticker)
                ticker_query = "select distinct(p_date) as ticker_date from " + self.table_name + " where ticker_id=" + (str)(ticker[0]) + " order by ticker_date;"
                self.cursor.execute(ticker_query)
                ticker_dates_list = [list(j) for j in self.cursor.fetchall()]
                date_data = None
                for ticker_date in ticker_dates_list:
                    discrete_columns = ""
                    if (not(self.drop_discrete_columns)):
                        discrete_columns = ", volume, number_of_trades"
                    ticker_date_query = "SELECT p_time, volume_weighted_average, open, close, high, low, time" + discrete_columns +" FROM " + self.table_name + " WHERE ticker_id=" + (str)(ticker[0]) + " and p_date = '" + ticker_date[0] + "' ORDER BY p_time;"
                    self.cursor.execute(ticker_date_query)
                    ticker_date_data_list = [list(k) for k in self.cursor.fetchall()]
                    row_data = list()
                    for ticker_date_data_row in ticker_date_data_list:
                        # avoid adding data that has zero records or less than 60% from a day (25.2/42 records, starting at 4:30 ending at 8:00)
                        if (len(ticker_date_data_list) > self.row_data_threshold):
                            ticker_date_data_row_0 = ticker_date_data_row[0].replace(":","")
                            row_data.append((ticker_date_data_row))
                        ## if impute
                        else:
                            if self.verbose == 2:
                                print(f'[DEBUG][PennyStockData]: dropping data ({ticker[0]}, {ticker_date[0]}) rows={len(ticker_date_data_list)} is < row_data_threshold: {self.row_data_threshold}')
                            break
                    
                        #if (len(row_data) > row_data_threshold):
                    row_data = np.array(row_data)
                    date_data[ticker_date[0]] = row_data
    
                        #with open(csv_filename, 'a', newline='') as file:
                        #    csvwriter = csv.writer(file)
                        #    csvwriter.writerow((ticker[0], ticker_date[0], row_data, row_label))
                        #file.close()
                    
                ticker_data[ticker[0]] = date_data
                #print(ticker_data)
            dataset = ticker_data
            #dataset = ticker_data
    
            if self.verbose >= 1:
                print(f'[INFO][PennyStockData]: Records have been numpied successfully on the variable dataset')
            return dataset


'''
