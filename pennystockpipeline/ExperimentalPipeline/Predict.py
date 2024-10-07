import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
import sqlite3

def predict(ticker, margin):
    SEQUENCE_LENGTH = 78  # One day of 5-min data
    FUTURE_STEPS = 58     # Predicting the remaining 58 time steps of the day
    VIEWFINDER = 14

    # Load the LSTM model
    model = load_model('lstm_model.h5')

    # Connect to SQLite database and query stock data
    db = 'stockdata.db'
    conn = sqlite3.connect(db)
    query = f"SELECT * FROM stockdata WHERE ticker = '{ticker}'"
    stock_data = pd.read_sql_query(query, conn)

    # Ensure data has at least SEQUENCE_LENGTH rows
    if len(stock_data) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough data for {ticker} (at least {SEQUENCE_LENGTH} rows needed).")

    # Extract relevant features for prediction (e.g., VWAP, volume, transactions)
    features = stock_data[['vwap', 'volume', 'transactions']].values

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features[-SEQUENCE_LENGTH:])

    # Reshape data for LSTM input: (1, SEQUENCE_LENGTH, num_features)
    input_data = np.reshape(scaled_features, (1, SEQUENCE_LENGTH, scaled_features.shape[1]))

    # Predict future values for the remaining time steps of the day
    input_data = input_data[:, :, 0]
    print(input_data.shape)
    predictions_scaled = model.predict(input_data)

    # Reverse the scaling for predicted VWAP prices
    predicted_vwap = scaler.inverse_transform(np.hstack((predictions_scaled[:, :, 0], np.zeros((predictions_scaled.shape[0], 2)))))[:, 0]

    # Get the current day's VWAP data (ground truth)
    current_vwap = stock_data['vwap'].iloc[-SEQUENCE_LENGTH:]

    # Plot actual vs predicted VWAP
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(SEQUENCE_LENGTH), current_vwap, label='Actual VWAP', color='blue')
    plt.plot(np.arange(SEQUENCE_LENGTH, SEQUENCE_LENGTH + FUTURE_STEPS), predicted_vwap, label='Predicted VWAP', color='orange')
    
    # Optionally, you can add more plotting features (buy/sell signals, etc.)
    
    plt.title(f'{ticker} VWAP Prediction for Remaining Day')
    plt.xlabel('Time Step')
    plt.ylabel('VWAP Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{ticker}_vwap_prediction.png')
    plt.show()

    # Close SQLite connection
    conn.close()

    return predicted_vwap

