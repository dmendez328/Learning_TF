import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Generate synthetic "stock" time series
np.random.seed(42)
time = np.arange(0, 1000)
trend = 0.05 * time
seasonal = 10 * np.sin(0.1 * time)
noise = np.random.normal(scale=2, size=len(time))
series = trend + seasonal + noise

# 2. Visualize the data
plt.figure(figsize=(10, 4))
plt.plot(time, series)
plt.title("Synthetic Stock Price Series")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

# 3. Create sequence-to-sequence dataset
def create_seq2seq_dataset(series, input_len, output_len):
    X, y = [], []
    for i in range(len(series) - input_len - output_len):
        X.append(series[i:i + input_len])
        y.append(series[i + input_len:i + input_len + output_len])
    return np.array(X), np.array(y)

input_steps = 30 # How many past steps to look at
output_steps = 5 # How many future steps to predict
X, y = create_seq2seq_dataset(series, input_steps, output_steps)

# 4. Reshape for LSTM (samples, timesteps, features)
# LSTM layers expect 3D input tensors of shape (sample, timesteps, features)
X = X[..., np.newaxis]
y = y[..., np.newaxis]

# 5. Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6. Build LSTM model
# Recall: return_sequences=True means it returns the full output sequence, its default is False
# All LSTM layers, except the last one, should be return_sequence=True
# If you want more LSTM layers, then keep return_sequence=True
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_steps, 1)), # (timesteps, features)
    tf.keras.layers.LSTM(64, return_sequences=True), # Return full sequence (for next LSTM)
    tf.keras.layers.LSTM(32, return_sequences=True), # Return full sequence (for next LSTM)
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(output_steps) # Predict next N steps
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 7. Train model
model.fit(X_train, y_train[..., 0], epochs=30, batch_size=32, validation_split=0.1)

# 8. Predict and plot
preds = model.predict(X_test)
true = y_test[..., 0]

plt.figure(figsize=(12, 5))
plt.plot(true[0], label='Actual')
plt.plot(preds[0], label='Predicted')
plt.title("Multi-step Forecasting (First Test Sample)")
plt.xlabel("Time Steps Ahead")
plt.ylabel("Price")
plt.legend()
plt.show()