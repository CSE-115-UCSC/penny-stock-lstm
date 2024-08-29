## #!/usr/bin/env python

## PennyStockLabeller
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

from pennystockpipeline.PennyStockData import PennyStockData

class PennyStockLabeller():
    def __init__(self, psd, verbose=0) -> None:
        self.verbose = verbose
        
        self.psd = psd
        self.scaler = MinMaxScaler(feature_range=(0,1))
    
    def labelize(self, columns_to_compare=["volume_weighted_average"], horizon = 0.1):
        unlabelled_data_raw = np.array(self.psd.data)
        number_of_records = len(unlabelled_data_raw)
        
        unlabelled_data_df = pd.DataFrame(unlabelled_data_raw, columns=self.psd.headers)
        
        columns_to_compare_str = [column_to_compare + "_h_str" for column_to_compare in columns_to_compare]
        labels_df_data = ["no_buy"] * number_of_records
        labels_df = pd.DataFrame(data = labels_df_data, columns = columns_to_compare_str)
    
        labels_cursor = 0
    
        for ticker_id in unlabelled_data_df['ticker_id'].unique():
            ticker_data_df = unlabelled_data_df[unlabelled_data_df['ticker_id'] == ticker_id].copy()
            
            for ticker_date in ticker_data_df['p_date'].unique():
                ticker_date_data_df = ticker_data_df[ticker_data_df['p_date'] == ticker_date].copy()
    
                buy_triggered = False    
                labels_df_cursor = 0
    
                for ctc in columns_to_compare:
                    if ctc in ticker_date_data_df.columns:
                        for m in ticker_date_data_df.index:
                            if m >= ticker_date_data_df.index.max():
                                break
                            
                            comparison_lambda = lambda h, x, d: np.insert((h < (np.float32(x) / np.float32(d))), 0, False)
                            comparison_truth_table = comparison_lambda(horizon, ticker_date_data_df.loc[m+1:, ctc], ticker_date_data_df.loc[m, ctc])
                            
                            labels_df_indices = ticker_date_data_df[:len(comparison_truth_table)].index[comparison_truth_table]
                            labels_df.loc[labels_df_indices, ctc+"_h_str"] = "buy"
                            
                            if (labels_df_indices.shape[0] > 0):
                                buy_triggered = True
                                ticker_date_data_df = None
                                break
                                
                        if buy_triggered:
                            ticker_date_data_df = None
                            break
            ticker_date_data_df = None
            
        labelled_data_df = pd.concat([unlabelled_data_df, labels_df], axis=1)        
        self.labelled_data_df = labelled_data_df

        return self

    def to_csv(self, filename):
        self.labelled_data_df.to_csv(filename)
        print(f'The labelled data has been stored in {filename} in the root directory')
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
