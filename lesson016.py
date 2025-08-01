import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# This script demonstrates how to build a more complex model with multiple hidden layers
# It generates synthetic spiral data . . . Same description as previous scripts

# Load spiral data (same as previous)
def generate_spiral_data(points_per_class=100, num_classes=3):
    X = []
    y = []
    for class_number in range(num_classes):
        ix = range(points_per_class * class_number, points_per_class * (class_number + 1))
        r = np.linspace(0.0, 1, points_per_class)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2
        X.extend(np.c_[r * np.sin(t), r * np.cos(t)])
        y.extend([class_number] * points_per_class)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# Generate data
X, y = generate_spiral_data()
y_onehot = tf.one_hot(y, depth=3)

# Build model with multiple hidden layers
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,), name='input_layer'),
    tf.keras.layers.Dense(16, activation='relu', name='hidden_1'),
    tf.keras.layers.Dense(16, activation='relu', name='hidden_2'),
    tf.keras.layers.Dense(3, activation='softmax', name='output_layer')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y_onehot, epochs=100, batch_size=32, verbose=0)

# Define intermediate model to extract hidden layer outputs
# Let's visualize hidden layer 1 and hidden layer 2 outputs
hidden_layer_1_model = tf.keras.Model(inputs=model.inputs,
                                      outputs=model.get_layer('hidden_1').output)
hidden_layer_2_model = tf.keras.Model(inputs=model.inputs,
                                      outputs=model.get_layer('hidden_2').output)

# Predict hidden features
features_1 = hidden_layer_1_model.predict(X)
features_2 = hidden_layer_2_model.predict(X)

# Project 16 focuses on visualizing activations
def plot_hidden_features(features, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(features[:, 0], features[:, 1], c=y, cmap="coolwarm", s=15)
    plt.title(title)
    plt.xlabel("Activation 1")
    plt.ylabel("Activation 2")
    plt.grid(True)
    plt.colorbar(label="Class")
    plt.show()

# Plot activations for both hidden layers
plot_hidden_features(features_1, "Hidden Layer 1 Activations")
plot_hidden_features(features_2, "Hidden Layer 2 Activations")
