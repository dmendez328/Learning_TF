import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# This script builds on lesson014.py to include validation and early stopping
# It generates synthetic spiral data, builds a model, trains it with validation, and visualizes the results

# Training loss can continue decreasing, but validation loss might increase - clear sign of overfitting
# Early stopping prevents this by halting training when performance stops improving on unseen data

# Check out lesson014.py, this is what we will build upon
def generate_spiral_data(points_per_class=100, num_classes=3):
    X = []
    y = []
    for class_number in range(num_classes):
        r = np.linspace(0.0, 1, points_per_class)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2
        X.extend(np.c_[r * np.sin(t), r * np.cos(t)])
        y.extend([class_number] * points_per_class)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# Generate data and split into train/val
X, y = generate_spiral_data()
y_onehot = tf.one_hot(y, depth=3)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_onehot.numpy(), test_size=0.2, random_state=42)

print(f"Training size: {X_train.shape}, Validation size: {X_val.shape}")

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping stops training when validation loss stops improving
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Recall: Callback is an object that performs actions during training
# Train model with validation
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop],
                    verbose=0)

# Final evaluation
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Final Validation Accuracy: {val_acc:.4f}")

# Plot loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
