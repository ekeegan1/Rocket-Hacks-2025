import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the data
data = pd.read_csv('aapl.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate features
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['MA_100'] = data['Close'].rolling(window=100).mean()

# Calculate MACD
def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

data['MACD'], data['Signal_Line'] = calculate_macd(data)

# Calculate 52-Week High/Low
data['52W_High'] = data['Close'].rolling(window=252).max()
data['52W_Low'] = data['Close'].rolling(window=252).min()
data['Above_52W_High'] = (data['Close'] > data['52W_High']).astype(int)
data['Below_52W_Low'] = (data['Close'] < data['52W_Low']).astype(int)

# Drop rows with NaN values
data.dropna(inplace=True)

# Select features for the model
features = ['Close', 'MA_10', 'MA_50', 'MA_100', 'MACD', 'Signal_Line', 'Above_52W_High', 'Below_52W_Low']
X = data[features].values
y = data['Close'].values

# Create sequences
sequence_length = 60
X_sequences, y_sequences = [], []
for i in range(sequence_length, len(X)):
    X_sequences.append(X[i-sequence_length:i])
    y_sequences.append(y[i])

X_sequences, y_sequences = np.array(X_sequences), np.array(y_sequences)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.1, random_state=42, shuffle = False)

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
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
y_pred = model.predict(X_test)

test_dates = data.index[-len(y_test):]

# Create a figure and axis object
# Create a figure and axis object for the line plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the Actual and Predicted prices as lines
ax.plot(test_dates, y_test, label='Actual Prices', color='blue', linestyle='-')  # Optional markers for clarity
ax.plot(test_dates, y_pred.flatten(), label='Predicted Prices', color='red', linestyle='--')

# Set the title and axis labels
ax.set_title('Predicted vs Actual Stock Prices (AAPL)', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Stock Price', fontsize=12)

# Format the x-axis to show readable dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())

# Rotate the date labels for better visibility
fig.autofmt_xdate()

# Add a legend to distinguish between lines
ax.legend(fontsize=10)

# Adjust layout for a better fit
plt.tight_layout()

# Display the plot
plt.show()


print(len(test_dates), len(y_test), len(y_pred))
