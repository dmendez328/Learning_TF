import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Generate synthetic time series (e.g. temperature-like sine wave)
time = np.arange(0, 500) # Domain 
series = np.sin(0.1 * time) + np.random.normal(scale=0.1, size=len(time)) # Function

# 2. Plot the generated series
plt.figure(figsize=(10, 4))
plt.plot(time, series)
plt.title("Synthetic Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# 3. Create sliding window dataset
def create_dataset(series, window_size):
    X, y = [], []
    # Break series into input-output pairs (input = fixed number of past values, output = the next value to predict)
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

window_size = 20
X, y = create_dataset(series, window_size)

# 4. Train/test split (80/20)
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 5. Build model (simple feedforward network)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 6. Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 7. Predict and plot
preds = model.predict(X_test).flatten()

plt.figure(figsize=(10, 4))
plt.plot(range(len(y_test)), y_test, label='Actual')
plt.plot(range(len(preds)), preds, label='Predicted')
plt.title("Forecast vs. Actual")
plt.legend()
plt.show()