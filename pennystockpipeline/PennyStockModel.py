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

from sklearn.preprocessing import MinMaxScaler

from pennystockpipeline.PennyStockData import PennyStockData

class PennyStockModel(nn.Module):
    # input_size : number of features in input at each time step
    # hidden_size : Number of LSTM units 
    # num_layers : number of LSTM layers 
    def __init__(self, input_size, hidden_size, num_layers, device='cpu') -> None: 
        super(PennyStockModel, self).__init__() #initializes the parent class nn.Module
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
        sequence_to_plot = self.x_test.squeeze().cpu().numpy()
    
        # Use the last 36 data points as the starting point
        historical_data = sequence_to_plot[-1]
        
        # Initialize a list to store the forecasted values
        forecasted_values = []
         
        # Use the trained model to forecast future values
        with torch.no_grad():
            for _ in range(num_forecast_steps * 2):
                # Prepare the historical_data tensor
                historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(self.device)
                # Use the model to predict the next value
                predicted_value = self(historical_data_tensor).cpu().numpy()[0, 0]
                
                # Append the predicted value to the forecasted_values list
                forecasted_values.append(predicted_value[0])
         
                # Update the historical_data sequence by removing the oldest value and adding the predicted value
                #historical_data
                historical_data = np.roll(historical_data, shift=-1)
                historical_data[-1] = predicted_value
    
        # Generate
        model_psd_data_df = pd.DataFrame(self.psd.data, columns=self.psd.headers)
        psd_ds_dates = model_psd_data_df['p_date'].values.tolist()
        psd_ds_times = model_psd_data_df['p_time'].values.tolist()
    
        #psd_ds_dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in psd_ds_dates]
        #last_date = max(psd_ds_dates_dt)
        #next_date = last_date + timedelta(days=1)
    
        next_date = self.psd.next_max_date
    
        next_date_ls = [next_date for i in range(num_forecast_steps*3)]
    
        #print(type(psd_ds_dates), type(psd_ds_dates[0]))
        #print(type(next_date_ls), type(next_date_ls[0]))
    
        psd_ds_dates = psd_ds_dates + next_date_ls
    
        #print(type(psd_ds_dates), type(psd_ds_dates[0]))
        #print(type(next_date_ls), type(next_date_ls[0]))
        
        time_steps_set = psd_ds_times[:num_forecast_steps]
        psd_ds_times = psd_ds_times + psd_ds_times[:num_forecast_steps*3]
    
        #print(type(psd_ds_dates[0]), psd_ds_dates[0])
        
        # Concatenate the original index with the future dates
        self.psd_ds_dates = psd_ds_dates
        self.psd_ds_times = psd_ds_times
    
        self.forecasted_values = forecasted_values
        self.sequence_to_plot = sequence_to_plot
    
        return self

    def create_dataloaders(self, psd, batch_size=16):
        # Create DataLoader for batch training
        self.batch_size = batch_size

        self.psd = psd
        
        self.x_train, self.y_train = psd.x_train, psd.y_train
        self.x_test, self.y_test = psd.x_test, psd.y_test
        
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

    def plot_training_test_loss(self):
        _x = np.linspace(1, self.num_epochs, self.num_epochs)
        plt.plot(_x, self.train_hist, scalex=True, label="Training loss")
        plt.plot(_x, self.test_hist, label="Test loss")
        plt.legend()
        plt.show()
        
        return self

    def plot_forecasting(self):

        forecasted_values = self.forecasted_values
        
        psd_ds_dates = self.psd_ds_dates[-100:]
        psd_ds_times = self.psd_ds_times[-100:]
    
        # last 100 rows
        psd_ds_datetimes = []
        
        [psd_ds_datetimes.append(d + " " + t) for d,t in zip(psd_ds_dates, psd_ds_times)]
    
        test_data_x = self.psd.x_test.squeeze().reshape(-1, 1).squeeze()
        #print(len(test_data_x))
        #print(type(test_data_x), test_data_x.shape)
        
        sequence_to_plot = self.sequence_to_plot
        
        model_psd_data_df = pd.DataFrame(self.psd.data, columns=self.psd.headers)
        
        #set the size of the plot 
        plt.rcParams['figure.figsize'] = [14, 4] 
        
        #Test data
        plt.plot(psd_ds_datetimes[-100:-40], test_data_x[-100:-40], label = "input", color = "b") 
        #reverse the scaling transformation
        original_cases = self.psd.scaler.inverse_transform(np.expand_dims(sequence_to_plot[-1], axis=0))#.flatten() 
        original_cases = original_cases.reshape(-1, 1).squeeze()
        
        #the historical data used as input for forecasting
        plt.plot(psd_ds_datetimes[-40:-20], original_cases, label='actual values', color='green') 
        
        #Forecasted Values 
        #reverse the scaling transformation
        forecasted_cases = self.psd.scaler.inverse_transform(np.expand_dims(forecasted_values, axis=0)).flatten() 
        # plotting the forecasted values
        plt.plot(psd_ds_datetimes[-40:], forecasted_cases, label='forecasted values', color='red') 
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
    
        plt.xticks(psd_ds_datetimes, psd_ds_datetimes, rotation='vertical')
        plt.locator_params(axis='x', nbins=len(psd_ds_datetimes)/5)
        plt.tight_layout(pad=4)
        plt.subplots_adjust(bottom=0.15)
    
        plt.legend()
        plt.title('Time Series Forecasting')
        plt.grid(True)

        return self
