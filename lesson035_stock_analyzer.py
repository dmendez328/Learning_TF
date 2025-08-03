import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' Generate synthetic "stock" time series data '''

# Generate data

np.random.seed(42) # For future reolication
time = np.arange(0, 1500) # The timestep range from a to b
trend = 0.05 * time # The linear upward trend over the times
seasonal = 10 * np.sin(0.1 * time) # A repeating sin wave, modeling seasonal patterns
noise = np.random.normal(scale=2, size=len(time)) # Add gaussian noise to simulate measurment errors
series = trend + seasonal + noise # Putting everything together

# Visualize the data via plot
plt.figure(figsize=(10, 4))
plt.plot(time, series)
plt.title("Synthetic Stock Price Series")
plt.xlabel("Time")
plt.ylabel("Price")
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

X = []

for i in range(len(df) - input_steps):
  window = df.iloc[i:i + input_steps].values # Shape = (30, 5)
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
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same'), # (Batch Size, 30, 32)

    # Layer 2
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'), # (Batch Size, 30, 64)
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same'), # (Batch Size, 15, 64)

    # LSTM Block

    # Layer 1
    tf.keras.layers.LSTM(units=128, return_sequences=True), # (Batch Size, 15, 128)

    # Layer 2
    tf.keras.layers.LSTM(units=64, return_sequences=True), # (Batch Size, 15, 64)

    # Flatten
    tf.keras.layers.Flatten(), # (Batch Size, 960)

    # Dense Block

    # Layer 1
    tf.keras.layers.Dense(units=256, activation='relu'), # (Batch Size, 256)

    # Layer 2
    tf.keras.layers.Dense(units=64, activation='relu'), # (Batch Size, 64)

    # Layer 3
    tf.keras.layers.Dense(units=16, activation='relu'), # (Batch Size, 16)

    # Output Layer (Predict next n timesteps)
    tf.keras.layers.Dense(units=output_steps) # (Batch Size, output_steps)

])

# Compile the model / Model summary
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
model.summary()

''' Train the model '''

split = int(0.8 * len(X))
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

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))