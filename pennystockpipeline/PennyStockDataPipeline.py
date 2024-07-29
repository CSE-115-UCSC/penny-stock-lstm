## #!/usr/bin/env python

## PennyStockDataPipeline
## This pipeline performs the following operations on data:
##  - Data Gathering
##  - Data Cleaning and Transformation
##  - (Later, mat be) Exploratory Data analysis (EDA)

## ToDo: Allow specifying features list to be selected from the database table or should we use Torch Filter

## Please refer to the PennyStockPipeline.md for details of this pipeline

import torch
from torch.utils.data import Dataset

# import torchdata.datapipes as dp
import csv, os, sys
import sqlite3
from time import time, strftime

class PennyStockDataPipeline(Dataset):
    
    def __init__(self, database_name_with_path, table_name):
        ## Extract
        assert (database_name_with_path and database_name_with_path.strip()) == ""
        assert (table_name and table_name.strip()) == ""
        
        self.path = file_exists(database_name_with_path)
        
        self.cursor = self.connect_db()
        self.table_name = table_name
        self.data, self.headers, self.size = self.get_data()

        ## Transform
        ## Use Torch Transforms to clean and formatting the data
        ## torchvision.transforms

        ## Load
        ## Use Torch Datasets
        ## To Do: Get input for load Filter
        
        
        
    # Connect to SQlite database
    def connect_db(self):
        cursor = None
        try:
          sqliteConnection = sqlite3.connect(self.path)
          cursor = sqliteConnection.cursor()
          print(f'SQlite connected with {db}')
        except:
          sys.stderr.write("Failed to connect to database")
        return cursor

    def get_data(self):
        data = []
        # Execute query
        try:
          query = "SELECT * FROM " + self.table_name
          self.cursor.execute(query)
            
          data = [list(i[0]) for i in self.cursor.fetchall()]
          headers = data.keys()
          size = len(data)
          
        except:
          sys.stderr.write("Failed to execute query")

        return data, headers, size

    # Private valid file checker
    def file_exists(path):
        if os.path.isfile(path):
            return path
        else:
            raise FileNotFoundError('{} is not found.'.format(path))

