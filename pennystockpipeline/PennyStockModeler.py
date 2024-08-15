## #!/usr/bin/env python

## PennyStockModelPipeline
## 
import csv, os, sys
import numpy as np
import pandas as pd
from time import time, strftime, gmtime
from datetime import datetime

import matplotlib.pyplot as plt # Visualization 
import matplotlib.dates as mdates # Formatting dates
import seaborn as sns # Visualization

import torch # Library for implementing Deep Neural Network 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from pennystockpipeline.PennyStockData import PennyStockData

class PennyStockModeler(nn.Module):
    # input_size : number of features in input at each time step
    # hidden_size : Number of LSTM units 
    # num_layers : number of LSTM layers 
    def __init__(self, input_size, hidden_size, num_layers, device='cpu') -> None: 
        super(PennyStockModeler, self).__init__() #initializes the parent class nn.Module
        self.device = device
        if device=='cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #.to(self.device)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x): # defines forward pass of the neural network       
        out, _ = self.lstm(x)
        out = self.linear(out)
        
        return out

    def forecast(self, num_forecast_steps):
        # Define the number of future time steps to forecast
        self.num_forecast_steps = num_forecast_steps
         
        # Convert to NumPy and remove singleton dimensions
        if self.device == 'cuda':
            sequence_to_plot = self.x_test.squeeze().cuda().numpy()
        else:
            sequence_to_plot = self.x_test.squeeze().cpu().numpy()
         
        # Use the last 36 data points as the starting point
        historical_data = sequence_to_plot[-1]
        print(historical_data.shape)
         
        # Initialize a list to store the forecasted values
        forecasted_values = []
         
        # Use the trained model to forecast future values
        with torch.no_grad():
            for _ in range(num_forecast_steps*2):
                # Prepare the historical_data tensor
                historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(self.device)
                # Use the model to predict the next value
                if self.device == 'cuda':
                    predicted_value = self(historical_data_tensor).cuda().numpy()[0, 0]
                else:
                    predicted_value = self(historical_data_tensor).cpu().numpy()[0, 0]
         
                # Append the predicted value to the forecasted_values list
                forecasted_values.append(predicted_value[0])
         
                # Update the historical_data sequence by removing the oldest value and adding the predicted value
                historical_data = np.roll(historical_data, shift=-1)
                historical_data[:-1] = predicted_value
        
        # Generate futute dates
        last_date = max(self.psd.ds_dates, key=lambda d: datetime.strptime(d, '%Y-%m-%d'))
        #last_date = np.transpose().max()
         
        # Generate the next 36 5-min interval times 
        future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=num_forecast_steps)
         
        # Concatenate the original index with the future dates
        combined_index = np.transpose(self.psd.ds_dates).index.append(future_dates)

        self.forecasted_values = forecasted_values
        self.combined_index = combined_index
        self.sequence_to_plot = sequence_to_plot
        
        return self
    

    def plot_training_test_loss(self):
        _x = np.linspace(1, self.num_epochs, self.num_epochs)
        plt.plot(_x, self.train_hist, scalex=True, label="Training loss")
        plt.plot(_x, self.test_hist, label="Test loss")
        plt.legend()
        plt.show()
        
        return self

    def split_dataset(self, psd, split=0.8, to_torch=True):
        ##
        self.psd = psd
        x, y = psd.xs, psd.ys

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

    def create_dataloaders(self, batch_size=16):
        # Create DataLoader for batch training
        self.batch_size = batch_size
        
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True) 
        
        # Create DataLoader for batch training
        self.test_dataset = TensorDataset(self.x_test, self.y_test)
        self.test_loader = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle=False)
        
        return self
        
    def train_model(self, loss_fn, optimizer, num_epochs=50):
        model = self
        #self.loss_fn = loss_fn
        #self.optimizer = optimizer
        
        self.num_epochs = num_epochs
        train_hist =[]
        test_hist =[]
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
         
            # Training
            model.train()
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = model(batch_x)
                #print(predictions)
                loss = loss_fn(predictions, batch_y)
         
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
         
                total_loss += loss.item()
         
            # Calculate average training loss and accuracy
            average_loss = total_loss / len(self.train_loader)
            train_hist.append(average_loss)
         
            # Validation on test data
            model.eval()
            with torch.no_grad():
                total_test_loss = 0.0
         
                for batch_x_test, batch_y_test in self.test_loader:
                    batch_x_test, batch_y_test = batch_x_test.to(self.device), batch_y_test.to(self.device)
                    predictions_test = model(batch_x_test)
                    test_loss = loss_fn(predictions_test, batch_y_test)
         
                    total_test_loss += test_loss.item()
         
                # Calculate average test loss and accuracy
                average_test_loss = total_test_loss / len(self.test_loader)
                test_hist.append(average_test_loss)
            if (epoch+1)%10==0:
                print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')

        self.train_hist = train_hist
        self.test_hist = test_hist
        
        return self
