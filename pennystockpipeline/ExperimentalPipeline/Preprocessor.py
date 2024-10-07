import sqlite3
import pandas as pd
import pandas_ta as ta
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler

class Preprocessor():

    def __init__(self, db_filepath, drift=False, volatility=False, moving_avg=False, momentum=False, volume_features=False, support_resistance=False, bollinger_bands=False, z_score=False):
        # Load the data from the database
        self.conn = sqlite3.connect(db_filepath)
        self.df = pd.read_sql_query("SELECT * FROM stockdata_imputed", self.conn).drop(columns='otc')
        
        # Preprocess based on flags
        if drift:
            self.drift()
        if volatility:
            self.volatility()
        if moving_avg:
            self.moving_averages()
        if momentum:
            self.momentum_indicators()
        if volume_features:
            self.volume_features()
        if support_resistance:
            self.support_resistance
        if bollinger_bands:
            self.bollinger_bands
        if z_score:
            self.z_score

        # Impute missing start of data
        self.df.bfill(inplace=True)

        self.save_to_db('historical_data/sd_pre.db')
        print("Modified preprocessed dataset with additional columns saved to historical_data/sd_pre.db and historical_data/sd_pre.csv")

    def drift(self):
        # Log returns for drift calculation
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
        # Calculate drift (mean of log returns over a rolling window)
        self.df['drift'] = self.df['log_return'].rolling(window=10).mean()

    def volatility(self):
        # Calculate rolling volatility (standard deviation of log returns)
        self.df['volatility'] = self.df['log_return'].rolling(window=10).std()

    def moving_averages(self):
        # Simple Moving Averages (SMA)
        self.df['SMA_5'] = self.df['close'].rolling(window=395).mean()
        self.df['SMA_50'] = self.df['close'].rolling(window=3950).mean()
        # Exponential Moving Averages (EMA)
        self.df['EMA_5'] = self.df['close'].ewm(span=395, adjust=False).mean()
        self.df['EMA_50'] = self.df['close'].ewm(span=3950, adjust=False).mean()

    def momentum_indicators(self):
        # Using pandas-ta for RSI and MACD (Moving average convergence divergence)
        self.df['RSI'] = ta.rsi(self.df['close'], length=1106)
        self.df['MACD'], self.df['MACD_signal'], self.df['MACD_hist'] = ta.macd(self.df['close'], fast=948, slow=2054, signal=711)
        self.df['MACD'], self.df['MACD_signal'], self.df['MACD_hist'] = ta.macd(self.df['close'], fast=948, slow=2054, signal=711).values.T

    def volume_features(self):
        # Volume Change
        self.df['volume_change'] = self.df['volume'].pct_change()
        # Volume Oscillator: Ratio of short-term vs long-term volume moving averages
        self.df['volume_sma_5'] = self.df['volume'].rolling(window=395).mean()
        self.df['volume_sma_50'] = self.df['volume'].rolling(window=3950).mean()
        self.df['volume_oscillator'] = (self.df['volume_sma_5'] - self.df['volume_sma_50']) / self.df['volume_sma_50']

    def support_resistance(self):
        # Calculate the pivot points
        self.df['Pivot'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Calculate support and resistance levels
        self.df['Resistance1'] = (2 * self.df['Pivot']) - self.df['low']
        self.df['Support1'] = (2 * self.df['Pivot']) - self.df['high']
        
        self.df['Resistance2'] = self.df['Pivot'] + (self.df['high'] - self.df['low'])
        self.df['Support2'] = self.df['Pivot'] - (self.df['high'] - self.df['low'])

    def bollinger_bands(self):
        # Bollinger Bands (20-day SMA (Simple Moving Average) with upper/lower bands at 2 standard deviations)
        self.df['SMA_20'] = self.df['close'].rolling(window=1580).mean()
        self.df['stddev_20'] = self.df['close'].rolling(window=1580).std()
        self.df['upper_band'] = self.df['SMA_20'] + (2 * self.df['stddev_20'])
        self.df['lower_band'] = self.df['SMA_20'] - (2 * self.df['stddev_20'])

    def z_score(self):
        # Z-score (standardized returns)
        self.df['rolling_mean'] = self.df['close'].rolling(window=20).mean()
        self.df['rolling_stddev'] = self.df['close'].rolling(window=20).std()
        self.df['z_score'] = (self.df['close'] - self.df['rolling_mean']) / self.df['rolling_stddev']

    def save_to_db(self, new_db_filepath):
        # Save the processed data into a new SQLite database
        conn = sqlite3.connect(new_db_filepath)
        self.df.to_sql('stockdata_processed', conn, if_exists='replace', index=False)
                # Save concatenated data for the ticker
        if not self.df.empty:
            self.df.to_csv(f'historical_data/sd_pre.csv', index=False)
        conn.close()
    
    def normalize(self):

        time_step = 79
        self.scalers = {}

        feature_columns = ['close', 'open', 'high', 'low', 'volume', 'vwap',
                    'SMA_5', 'SMA_50', 'EMA_5', 'EMA_50', 'volume_change',
                    'volume_sma_5', 'volume_sma_50', 'volume_oscillator',
                    'transactions', 'log_return', 'drift', 'volatility',
                    'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
        
        # List to store prepared data
        X_all, y_all = [], []

        self.scaled_dataframe = pd.DataFrame(columns=feature_columns)

        # Process each ticker individually
        for ticker in self.df['ticker'].unique():

            print(f"    Normalizing data for {ticker}. . .")
            # Filter the data for the current ticker
            ticker_data = self.df[self.df['ticker'] == ticker].copy()
            
            # Initialize a MinMaxScaler for the current ticker and store it
            self.scalers[ticker] = {}

            scaled_features = []

            # Scale each feature independently
            for feature in feature_columns:

                if feature != 'pst' and feature != 'ticker':
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

            self.scaled_dataframe = pd.concat([self.scaled_dataframe, ticker_data], axis=0, ignore_index=True)

        # Save as new database
        conn = sqlite3.connect('historical_data/sd_pre_n.db')
        self.scaled_dataframe.to_sql('sd_pre_n', conn, if_exists='replace', index=False)
        self.scaled_dataframe.to_csv('historical_data/sd_pre_n.csv', index=False)

        
        self.X = np.array(X_all)
        self.y = np.array(y_all)
        return self.scaled_dataframe, self.X, self.y