import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class StockLSTMModel:
    def __init__(self, db_filepath):
        self.db_filepath = db_filepath
        self.scalers = {}  # Dictionary to store scalers for each ticker

    def load_data(self):
        conn = sqlite3.connect(self.db_filepath)
        df = pd.read_sql_query("SELECT * FROM stockdata_processed", conn)
        self.tickers = df['ticker'].unique()
        conn.close()
        return df

    def prepare_data(self, df, time_step=78):
        print("inside prepare")
        # Define the columns to be scaled
        feature_columns = ['close', 'high', 'low', 'open', 'volume', 'vwap', 
                           'log_return', 'drift', 'volatility', 'SMA_5', 'SMA_50', 
                           'EMA_5', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 
                           'MACD_hist', 'volume_change', 'volume_sma_5', 'volume_sma_50', 
                           'volume_oscillator']
        
        # List to store prepared data
        X_all, y_all = [], []

        # Process each ticker individually
        for ticker in df['ticker'].unique():

            print(f"    Preparing data for {ticker}. . .")
            # Filter the data for the current ticker
            ticker_data = df[df['ticker'] == ticker].copy()
            
            # Initialize a MinMaxScaler for the current ticker and store it
            self.scalers[ticker] = {}
            
            scaled_features = []
            
            # Scale each feature independently
            for feature in ['close', 'open', 'high', 'low', 'volume', 'vwap', 'SMA_5', 'SMA_50', 'EMA_5', 'EMA_50', 'volume_change', 'volume_sma_5', 'volume_sma_50', 'volume_oscillator', 'transactions', 'log_return', 'drift', 'volatility', 'RSI', 'MACD']:
                scaler = MinMaxScaler(feature_range=(0, 1))
                ticker_data[feature] = scaler.fit_transform(ticker_data[feature].values.reshape(-1, 1))
                self.scalers[ticker][feature] = scaler  # Store the scaler for this feature
                
                scaled_features.append(ticker_data[feature].values)

            # Convert the scaled features into a numpy array
            scaled_features = np.stack(scaled_features, axis=-1)  # Shape: (n_samples, n_features)

            # Create time-step sequences
            for i in range(time_step, len(scaled_features)):
                X_all.append(scaled_features[i-time_step:i])
                y_all.append(ticker_data['close'].values[i])  # Use 'close' as the target for now
        
        return np.array(X_all), np.array(y_all)

    def create_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self):
        print("Loading data. . .")
        self.df = self.load_data()
        print("Preparing data with initial df shape:", self.df.shape, ". . .")
        X, y = self.prepare_data(self.df)
        print("Finished preparing the data. Final shape is-- X:", X.shape, "y:", y.shape)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Use early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Create and train the model
        print("Creating model. . .")
        self.model = self.create_model((X.shape[1], 1))
        print("Fitting model. . .")
        self.model.fit(X, y, epochs=50, batch_size=32)
        # Train the model with early stopping
        self.model.fit(
            X, y,
            epochs=50,
            callbacks=[early_stopping]  # Add the early stopping callback
        )
            
        # Save the model and associate it with the ticker
        print("Saving model. . .")
        self.model.save('lstm_model.h5')
        print("Finished.")


# # Usage:
# stock_lstm = StockLSTMModel('historical_data/sd_pre.db')
# stock_lstm.train_models()

# # Predicting for a specific ticker
# data_for_prediction = np.array([1.23, 1.24, 1.25])  # Example recent data
# prediction = stock_lstm.predict('AAPL', data_for_prediction)
# print(prediction)
