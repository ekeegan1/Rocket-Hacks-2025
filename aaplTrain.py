# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 10:54:11 2025

@author: logan
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.dates as mdates
from tensorflow.keras.optimizers import Adam

# Load the datase
file_path = 'aapl.csv'
data = pd.read_csv(file_path)

# Use only the 'Close' price for simplicity
close_prices = data['Close'].values.reshape(-1, 1)
data.set_index('Date', inplace = True)

# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create sequences for LSTM (e.g., use 60 days to predict the next day)
sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])  # Past 60 days
    y.append(scaled_data[i])  # Next day's price

X, y = np.array(X), np.array(y)

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle = False)




num_epochs = 5  # Replace 50 with the variable storing your epoch count

# Build the LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(100, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=num_epochs,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Plot training and validation loss over epochs
import matplotlib.pyplot as plt
plt.title(f'Training and Validation Loss (Trained for {num_epochs} Epochs)')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()



# Evaluate on test set
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")



test_dates = data.index[-len(y_test):]  # Align dates with y_test and predictions

test_dates = pd.to_datetime(test_dates)




# Make predictions on test data
predicted_prices = model.predict(X_test)

# Rescale predictions back to original scale
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(test_dates,actual_prices, label='Actual Prices', color='blue')
plt.plot(test_dates,predicted_prices, label='Predicted Prices', color='red')
plt.title(f'Actual vs Predicted Prices (Trained for {num_epochs} Epochs)')
plt.xlabel('Date')
plt.ylabel('Price')

# Format x-axis to show readable dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format as Year-Month-Day
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically place date ticks

# Rotate and align the date labels for better readability
plt.gcf().autofmt_xdate()

# Add the legend
plt.legend()

# Show the plot
plt.show()



