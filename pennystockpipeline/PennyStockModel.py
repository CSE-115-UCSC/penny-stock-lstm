## #!/usr/bin/env python

## PennyStockModelPipeline
## 
import os
#import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt # Visualization

import torch # Library for implementing Deep Neural Network 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score


from pennystockpipeline.PennyStockData import PennyStockData

class PennyStockModel(nn.Module):
    # input_size : number of features in input at each time step
    # hidden_size : Number of LSTM units 
    # num_layers : number of LSTM layers
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2, device='cpu') -> None:
        super(PennyStockModel, self).__init__()
        
        self.device = device
        if device=='cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

    def model_evaluation(self):

        # Evaluate the model and calculate RMSE and R² score
        self.eval()
        with torch.no_grad():
            test_predictions = []
            for batch_X_test in self.x_test:
                batch_X_test = batch_X_test.to(self.device).unsqueeze(0)  # Add batch dimension
                test_predictions.append(self(batch_X_test).to(self.device).numpy().flatten()[0])
        
        test_predictions = np.array(test_predictions)
        
        # Calculate RMSE and R² score
        rmse = np.sqrt(mean_squared_error(self.y_test.to(self.device).numpy(), test_predictions))
        r2 = r2_score(self.y_test.to(self.device).numpy(), test_predictions)
        
        print(f'RMSE: {rmse:.4f}')
        print(f'R² Score: {r2:.4f}')

        return self

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
            for _ in range(num_forecast_steps):
                # Prepare the historical_data tensor
                historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(self.device)
                # Use the model to predict the next value
                predicted_value = self(historical_data_tensor).cpu().numpy()[0, 0]
                
                # Append the predicted value to the forecasted_values list
                forecasted_values.append(predicted_value)
         
                # Update the historical_data sequence by removing the oldest value and adding the predicted value
                #historical_data
                historical_data = np.roll(historical_data, shift=-1)
                historical_data[-1] = predicted_value

        #last_date = test_data.index[-1]
        #future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=30)
        
        # Generate
        model_psd_data_df = pd.DataFrame(self.psd.data, columns=self.psd.headers)
        psd_ds_dates = model_psd_data_df['p_date'].values.tolist()
        psd_ds_times = model_psd_data_df['p_time'].values.tolist()
    
    
        next_date = self.psd.next_max_date
    
        next_date_ls = [next_date for i in range(num_forecast_steps)]
    
        psd_ds_dates = psd_ds_dates + next_date_ls
        
        #time_steps_set = ['16:30', '16:35', '16:40', '16:45', '16:50', '16:55', '17:00', '17:05', '17:10', '17:15', '17:20', '17:25', '17:30', '17:35', '17:40', '17:45', '17:50', '17:55', '18:00', '18:05']
        time_steps_set = ['16:30', '16:35', '16:40', '16:45', '16:50', '16:55', '17:00', '17:05', '17:10', '17:15', '17:20', '17:25', '17:30', '17:35', '17:40', '17:45', '17:50', '17:55', '18:00', '18:05', '18:10', '18:15', '18:20', '18:25', '18:30', '18:35', '18:40', '18:45', '18:50', '18:55', '19:00', '19:05', '19:10', '19:15', '19:20', '19:25', '19:30', '19:35', '19:40', '19:45', '19:50', '19:55', '20:00', '20:05', '20:10', '20:15', '20:20', '20:25' ]
        psd_ds_times = psd_ds_times + time_steps_set
        
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
        
        self.num_epochs = num_epochs
        train_hist =[]
        test_hist =[]

        #Initialize Variables for EarlyStopping
        #best_loss = float(1.0)
        #best_model_weights = None
        #patience = 10

        best_loss = float(100000)
        best_model_weights = None
        
        # Training loop
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            total_loss = 0.0
            # Training
            self.train()
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = self(batch_x)
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
            self.eval()
            with torch.no_grad():
                total_test_loss = 0.0
         
                for batch_x_test, batch_y_test in self.test_loader:
                    batch_x_test, batch_y_test = batch_x_test.to(self.device), batch_y_test.to(self.device)
                    predictions_test = self(batch_x_test)
                    test_loss = loss_fn(predictions_test, batch_y_test)
         
                    total_test_loss += test_loss.item()
         
                # Find best test loss for keepsake
                if (total_test_loss < best_loss):
                    print(f'best_loss: {total_test_loss} @ epoch: {epoch}')
                    best_loss = total_test_loss
                    best_model_weights = self.state_dict()

                # Calculate average test loss and accuracy
                average_test_loss = total_test_loss / len(self.test_loader)
                test_hist.append(average_test_loss)
                
                # Early stopping
                #if total_test_loss < best_loss:
                #    best_loss = total_test_loss
                #    best_model_weights = copy.deepcopy(self.state_dict())  # Deep copy here      
                #    patience = 10  # Reset patience counter
                #else:
                #    patience -= 1
                #    if patience == 0:
                #        break
                
            if (epoch+1)%10==0:
                print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.6f}, Test Loss: {average_test_loss:.6f}')

        self.train_hist = train_hist
        self.test_hist = test_hist

        self.load_state_dict(best_model_weights)

        return self

    def plot_training_test_loss(self):
        _x = np.linspace(1, self.num_epochs, self.num_epochs)
        plt.plot(_x, self.train_hist, scalex=True, label="Training loss")
        plt.plot(_x, self.test_hist, label="Test loss")
        plt.legend()
        plt.show()
        
        return self

    def save_model(self):
        torch.save(self.state_dict(), 'model_weights_PennyStockModel.pth')
        
        return self

    def load_model(self):
        if (self.__file_exists('model_weights_PennyStockModel.pth')):
            self.load_state_dict(torch.load('model_weights_PennyStockModel.pth', weights_only=False, map_location=torch.device('cpu')))
            self.eval()
        
        return self

    def plot_forecasting(self):
        forecasted_values = self.forecasted_values
        
        psd_ds_dates = self.psd_ds_dates[-100:]
        psd_ds_times = self.psd_ds_times[-100:]
    
        # last 100 rows
        psd_ds_datetimes = []
        
        [psd_ds_datetimes.append(d + " " + t) for d,t in zip(psd_ds_dates, psd_ds_times)]
        test_data_x = np.float32(self.psd.scaler.inverse_transform(self.x_test.squeeze()).reshape(-1, 1).squeeze())
        
        sequence_to_plot = self.sequence_to_plot
        
        model_psd_data_df = pd.DataFrame(self.psd.data, columns=self.psd.headers)
        
        #set the size of the plot 
        plt.rcParams['figure.figsize'] = [14, 4] 
        
        #Test data
        plt.plot(psd_ds_datetimes[-100:-20], test_data_x[-80:], label = "input", color = "b") 
        
        #reverse the scaling transformation
        original_cases = self.psd.scaler.inverse_transform(np.expand_dims(sequence_to_plot[-1], axis=0))#.flatten() 
        original_cases = original_cases.reshape(-1, 1).squeeze()
        
        #the historical data used as input for forecasting
        plt.plot(psd_ds_datetimes[-40:-20], original_cases, label='actual values', color='green') 
        
        #Forecasted Values 
        #reverse the scaling transformation
        #forecasted_cases = self.psd.scaler.inverse_transform(np.expand_dims(forecasted_values, axis=0)).flatten() 
        forecasted_cases = self.psd.scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1)).flatten()
        # plotting the forecasted values
        plt.plot(psd_ds_datetimes[-40:], forecasted_cases, label='forecasted values', color='red') 
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
    
        plt.xticks(psd_ds_datetimes, psd_ds_datetimes, rotation='vertical')
        plt.locator_params(axis='x', nbins=len(psd_ds_datetimes)/3)
        plt.tight_layout(pad=4)
        plt.subplots_adjust(bottom=0.15)
    
        plt.legend()
        plt.title('Time Series Forecasting')
        plt.grid(True)

        print(f'original_cases:')
        print(f'{original_cases}')

        print(f'forecasted_cases:')
        print(f'{forecasted_cases}')

        return self

    # Private valid file checker
    def __file_exists(self, path) -> bool:
        if os.path.isfile(path):
            return True
        else:
            print("[INFO][PennyStockModel]:", path, "not found")
            return False

    def model_evaluation(self):
        # Evaluate the model and calculate RMSE and R² score
        self.eval()
        with torch.no_grad():
            test_predictions = []
            for batch_X_test in self.x_test:
                batch_X_test = batch_X_test.to(self.device).unsqueeze(0)  # Add batch dimension
                test_predictions.append(self(batch_X_test).cpu().numpy().flatten()[0])

        test_predictions = np.array(test_predictions)

        # Calculate RMSE and R² score
        rmse = np.sqrt(mean_squared_error(self.y_test.cpu().numpy(), test_predictions))
        r2 = r2_score(self.y_test.cpu().numpy(), test_predictions)

        print(f'RMSE: {rmse:.4f}')
        print(f'R² Score: {r2:.4f}')

        return self

    '''
    
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0, device='cpu') -> None:
        super(PennyStockModel, self).__init__() #initializes the parent class nn.Module
        self.device = device
        if device=='cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True) #.to(self.device)
        if (dropout > 0):
            self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x): # defines forward pass of the neural network       
        out, _ = self.lstm(x)
        out = self.linear(out)
        
        return out
    '''