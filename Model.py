import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.models import load_model
import sys


class StockLSTMModel:
    def __init__(self, data, X, y, scalers):
        self.data = data
        self.X = X
        self.y = y
        self.scalers = scalers

    def create_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self):

        print("Shape of input sequencess is:", self.X.shape)

        # Use early stopping
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        # Create and train the model
        print("Creating model. . .")
        self.model = self.create_model()
        print("Fitting model. . .")
        self.model.fit(
            self.X, self.y,
            epochs=50,
            batch_size=32,
            validation_split=0.1,  # Use 10% of the data for validation
            callbacks=[early_stopping]
        )

        # Save the model and associate it with the ticker
        print("Saving model. . .")
        self.model.save('lstm_model.h5')
        print("Finished.")

    def predict(self, normalized_data, scalers, prediction_date, known_timestamps, horizon):
        # Load the pre-trained model
        print("Loading model...")
        model = load_model('lstm_modelone.h5')

        # Reconfigure layers
        model.add(LSTM(name="new", units=50, input_shape=(None, 23)))  # Match 23 features in input data

        # Recompile
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Convert 'pst' column to datetime if it's not already
        normalized_data['pst'] = pd.to_datetime(normalized_data['pst'])

        # Then, when filtering for the prediction date:
        prediction_date = pd.to_datetime(prediction_date).date()  # Ensure prediction_date is a datetime object


        # Filter the data for the prediction date
        print(f"Fetching data for {prediction_date}...")
        test_data = normalized_data[normalized_data['pst'].dt.date == prediction_date]
        
        if len(test_data) < known_timestamps:
            raise ValueError(f"Not enough data for {prediction_date} with {known_timestamps} timestamps.")

        # Get the known part of the day (from the beginning to the known timestamp)
        known_data = test_data[:known_timestamps]

        # Reshape known data to match model input (1, known_timestamps, n_features)
        X_known = known_data.drop(columns=['pst', 'ticker']).values
        X_known = X_known.reshape(1, known_timestamps, X_known.shape[-1])
        
        print(f"Predicting from timestamp {known_timestamps} to 79...")
        
        # Predicting the rest of the day (from known_timestamps to 79)
        predicted_sequence = []
        print("X_known's shape is", X_known.shape)
        input_sequence = X_known.copy()  # Use known data to initialize predictions
        # Ensure input_sequence is filled up to 78 time steps
        input_sequence = np.zeros((1, 78, 23))  # Create a placeholder for 78 time steps
        # Fill the last 50 time steps with your data
        input_sequence[0, -50:, :] = X_known  # Assuming X_known is (1, 50, 23)
        print("input_sequence's shape is", input_sequence.shape)
        for i in range(known_timestamps, 79):
            print("Loop i==", i)
            print("Predicting value. . .")
            # print(model.summary())
            predicted_value = model.predict(input_sequence)
            print("Predicted value is", predicted_value)
            predicted_sequence.append(predicted_value[0, 0])  # Append prediction (single output)
            
            # Shift input to include this prediction and remove the oldest timestamp
            input_sequence = np.concatenate((input_sequence[:, 1:, :], 
                                            predicted_value.reshape(1, 1, -1)), axis=1)
        
        # Inverse scale the known data and predictions for comparison
        scaler_close = scalers['close']  # Retrieve the scaler for 'close' price
        known_close_unscaled = scaler_close.inverse_transform(known_data['close'].values.reshape(-1, 1))
        predicted_unscaled = scaler_close.inverse_transform(np.array(predicted_sequence).reshape(-1, 1))
        
        # Combine known and predicted into one array for plotting
        full_day_prediction = np.concatenate((known_close_unscaled, predicted_unscaled), axis=0)

        # Plot the results
        plt.figure(figsize=(10, 6))
        
        # Plot known values
        plt.plot(range(known_timestamps), known_close_unscaled, label="Known Data", color='blue')
        
        # Plot predicted values
        plt.plot(range(known_timestamps, 79), predicted_unscaled, label="Predicted Data", color='red')
        
        # Plot full day (for comparison)
        true_full_day = scaler_close.inverse_transform(test_data['close'].values.reshape(-1, 1))
        plt.plot(range(79), true_full_day, label="True Full Day", color='green', linestyle='dashed')
        
        plt.xlabel("Timestamp")
        plt.ylabel("Close Price")
        plt.title(f"Predicted vs Actual for {prediction_date}")
        plt.legend()
        plt.show()
        
        # Check if the prediction hits the horizon
        horizon_reached = predicted_unscaled[-1] >= horizon
        print(f"Horizon reached: {horizon_reached}")
        return horizon_reached
