import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# This script demonstrates how to handle overfitting and underfitting in a regression task using TensorFlow and Keras.

# Step 1: Generate noisy sine data
np.random.seed(42) # Randome seed for reproducibility
x = np.linspace(-2 * np.pi, 2 * np.pi, 300)
print("x:\n", x)
y = np.sin(x) + np.random.normal(scale=0.2, size=len(x)) # Adding noise to the sine wave
print("\ny:\n", y)

# Reshape to feed into model
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

print("\nx:\n", x)
print("\ny:\n", y)

# Split into train and validation sets
x_train, x_val = x[:200], x[200:]
y_train, y_val = y[:200], y[200:]

# Step 2: Define a model builder for testing different architectures
def build_model(hidden_units=20, regularizer=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_units, activation='relu', kernel_regularizer=regularizer),
        tf.keras.layers.Dense(hidden_units, activation='relu', kernel_regularizer=regularizer),
        tf.keras.layers.Dense(1)
    ])

# Step 3: Train model and track history
def train_model(model, title):
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=300,
        verbose=0
    )

    # Plot training and validation loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss Curve: {title}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

# Step 4: Try different settings

# 1. Underfitting: too few hidden units
underfit_model = build_model(hidden_units=2)
train_model(underfit_model, "Underfitting (2 hidden units)")

# 2. Overfitting: too many units, no regularization
overfit_model = build_model(hidden_units=100)
train_model(overfit_model, "Overfitting (100 hidden units)")

# 3. Just right: moderate model
just_right = build_model(hidden_units=20)
train_model(just_right, "Balanced (20 hidden units)")

# 4. Regularization (L2)
l2_model = build_model(hidden_units=100, regularizer=tf.keras.regularizers.l2(0.01))
train_model(l2_model, "L2 Regularization")

# 5. Early stopping
early_model = build_model(hidden_units=100)
early_model.compile(optimizer='adam', loss='mse')
callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = early_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=300,
    callbacks=[callback],
    verbose=0
)

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Early Stopping")
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()
