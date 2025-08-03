import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Note:
# Stock market is open ~ 250 days a year
# Market is open for 6.5 hours; on the 15 min timeframe, 26 bars
# For 5 years, open market 250 days a year for 6.5 hours, 15 min timeframe (26 bars a day) -> 32500 bars
# Most historical price APIs have data for companies going back 5 years

''' Generate synthetic "stock" time series data '''

# Generate data

np.random.seed(42)

# 1. Time
time = np.arange(0, 6000)

# 2. Exponential base trend
base_price = 50  # starting price
growth_rate = 0.0006  # tweakable
trend = base_price * np.exp(growth_rate * time)

# 3. Seasonal noise (optional)
seasonal = 10 * np.sin(0.1 * time)

# 4. Gaussian noise
noise = np.random.normal(scale=2, size=len(time))

# 5. Event: long drawdowns (corrections)
event_noise = np.zeros(len(time))

num_downtrends = 35
max_duration = 150

for _ in range(num_downtrends):
    start = np.random.randint(0, len(time) - max_duration)
    duration = np.random.randint(40, max_duration) # Change this to make the downtrends last longer
    slope = -np.random.uniform(1, 9) # Change this to make the downtrend slopes more dramatic (Bigger Drops)

    for t in range(duration):
        event_noise[start + t] += slope * t  # gradual drawdown

# 6. Final synthetic series
series = trend + seasonal + noise + event_noise

# 7. Plot
plt.figure(figsize=(14, 5))
plt.plot(time, series)
plt.title("Synthetic Stock Series: Exponential Growth + Corrections (MSFT-style)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# Create sequence-tosequence dataset
def create_seq_to_seq_data(series, input_len, output_len):
  x, y = [], []
  series_copy = series.copy()
  series_copy = series.tolist()

  for i in range(len(series_copy)):
    x.append(series_copy[i])
  for i in range(len(series_copy) - 1):
    y.append(series_copy[i + 1])

  return x, y

input_steps = 30 # How many past steps to look at
output_steps = 1 # The number of future steps to predict
x, y = create_seq_to_seq_data(series, input_steps, output_steps)

# Example of what the values look like
print("\nx (First 3):", x[:3])
print("\ny (First 3):", y[:3])

print("x Length:", len(x), "\ny Length:", len(y), "\nseries Length:", len(series))

''' Perform analysis on stock close prices '''

# Analytical functions
def sma(price_list):
  sma = []
  for i in range(input_steps - 1, len(price_list)):
    total = 0
    for u in range(0, input_steps):
      total = total + price_list[i - u]
    mean = total / float(input_steps)
    sma.append(mean)
  return sma

def ema(price_list, input_steps, smoothing=2):
    ema_values = []
    multiplier = smoothing / (1 + input_steps)

    # Start with a simple average for the first EMA value
    initial_sum = sum(price_list[:input_steps])
    prev_ema = initial_sum / float(input_steps)
    ema_values.append(prev_ema)

    # Continue calculating EMA for the rest
    for price in price_list[input_steps:]:
        current_ema = (price - prev_ema) * multiplier + prev_ema
        ema_values.append(current_ema)
        prev_ema = current_ema

    return ema_values

def bollinger_bands(price_list, input_steps, num_std=2):
    sma_list = []
    upper_band = []
    lower_band = []

    for i in range(input_steps - 1, len(price_list)):
        # Get the current window of prices
        window = []
        for u in range(0, input_steps):
            window.append(price_list[i - u])

        mean = sum(window) / float(input_steps)

        # Calculate standard deviation manually
        squared_diffs = [(p - mean) ** 2 for p in window]
        variance = sum(squared_diffs) / float(input_steps)
        std_dev = variance ** 0.5

        # Store results
        sma_list.append(mean)
        upper_band.append(mean + num_std * std_dev)
        lower_band.append(mean - num_std * std_dev)

    return upper_band, lower_band

# Calling functions and getting lists

price_list = series.copy()

# Make numpy arrays into regular arrays
price_list = price_list.tolist()

for i in range(input_steps):
  price_list.pop(0)
  y.pop(0)

# Call analytical functions
sma_list = sma(series)
ema_list = ema(series, input_steps)
upper_band, lower_band = bollinger_bands(series, input_steps)

# Alter lists to make a them match in length
sma_list.pop(0)
ema_list.pop(0)
upper_band.pop(0)
lower_band.pop(0)

print("Price:", len(price_list))
print("SMA:", len(sma_list))
print("EMA:", len(ema_list))
print("Upper Band:", len(upper_band))
print("Lower Band:", len(lower_band))

''' Create dataframes '''

# Building feature dataframe
df = pd.DataFrame({
    'Price': price_list,
    'SMA': sma_list,
    'EMA': ema_list,
    'Upper Band': upper_band,
    'Lower Band': lower_band
})

print(len(df))
print(df.head())

# Normalizing dataframe

# Normalization scalars have to be different
df_scalar = MinMaxScaler() # Type of normalization, only for df
df_scaled = df_scalar.fit_transform(df) # Normalize it
df_scaled = pd.DataFrame(df_scaled, columns=df.columns) # Turn from numpy to pandas, because type converted after normalization

y_scalar = MinMaxScaler() # Type of normalization, only for y
y = y_scalar.fit_transform(np.array(y).reshape(-1, 1)) # Normalize y

print(len(df_scaled))
print(df_scaled.head())
print(type(df_scaled))

X = []

for i in range(len(df_scaled) - input_steps):
  window = df_scaled.iloc[i:i + input_steps].values # Shape = (30, 5)
  X.append(window)

if len(y) > len(X):
  # Slice y to match X
  y = np.array(y[:len(X)])
elif len(y) < len(X):
  # Slice X to match y
  X = np.array(X[:len(y)])

print(len(X))
print(len(y))

''' Model construction '''

# Build model

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(input_steps, 5)), # (Batch Size, 30, 5)

    # CNN Block

    # Layer 1
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'), # (Batch Size, 30, 32)
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same'), # (Batch Size, 30, 32)

    # Layer 2
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'), # (Batch Size, 30, 64)
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same'), # (Batch Size, 30, 64)

    # Layer 3
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'), # (Batch Size, 15, 128
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same'), # (Batch Size, 15, 128)

    # LSTM Block

    # Layer 1
    tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(units=128, return_sequences=True)
    ), # (Batch Size, 15, 256) [256 because 128 units in this LSTM, back and forth]

    # Layer 2
    tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(units=64, return_sequences=False)
    ), # (Batch Size, 15, 128) [128 because 64 units in this LSTM, back and forth]

    # Dense Block

    # Layer 1
    tf.keras.layers.Dense(units=64, activation='relu'), # (Batch Size, 64)

    # Layer 2
    tf.keras.layers.Dense(units=32, activation='relu'), # (Batch Size, 32)

    # Layer 3
    tf.keras.layers.Dense(units=16, activation='relu'), # (Batch Size, 16)

    # Output Layer (Predict next n timesteps)
    tf.keras.layers.Dense(units=output_steps) # (Batch Size, output_steps)

])

# Compile the model / Model summary
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

''' Train the model '''

# Split the data to training and testing, 80% training
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(type(X_train))
print(type(y_train))
print(type(X_test))
print(type(y_test))

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
