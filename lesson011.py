import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# This script helps understand how to visualize weights and activations

'''
Step 1: Load and preprocess data
'''

iris = load_iris()
X = iris.data
Y = iris.target
print("\nX:", X, "\nY:", Y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled Features:\n", X_scaled)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

'''
Step 2: Build model with Functional API to access intermediate layers
'''

# Input Layer: 4 features (Iris dataset has 4 features)
inputs = tf.keras.Input(shape=(4,))
# Hidden Layer: 10 neurons with ReLU activation, the number of 10 neurons is arbitrary but adds complexity and non-linearity
hidden = tf.keras.layers.Dense(10, activation='relu', name="hidden_layer")(inputs)
# Output Layer: 3 classes (Iris dataset has 3 classes), using softmax for multi-class classification
outputs = tf.keras.layers.Dense(3, activation='softmax')(hidden)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=30, validation_split=0.2)

'''
Step 3: Visualize Weights of First Hidden Layer
'''

# Get weights and biases of the hidden layer
# The weights shape will be (input_dim, hidden_units) and biases shape will be (hidden_units,)
weights, biases = model.get_layer("hidden_layer").get_weights()
print("\nHidden Layer Weights:\n", weights)
print("\nHidden Layer Biases:\n", biases)

# The figure shows a heatmap of the weights of the hidden layer
# Rows are the input features, columns are the hidden neurons
# The color intensity represents the weight magnitude
# Bright Yellow -> High Positive Weight; Dark Purple -> High Negative Weight; Green -> Near Zero Weight (Little Influence)
print("\nEnter Figure 1 . . .")
plt.figure(figsize=(10, 4))
plt.imshow(weights, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Hidden Layer Weights (shape: [input_dim, hidden_units])")
plt.xlabel("Hidden Neuron Index")
plt.ylabel("Input Feature Index")
plt.show()

'''
Step 4: Visualize Activations for a Few Test Samples
'''

# Create a model that outputs activations of the hidden layer
# The model takes the same input as the original model but outputs the hidden layer's activations
# This allows us to see how the hidden layer responds to different inputs
activation_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("hidden_layer").output)

# Pick a few test examples
sample_inputs = X_test[:5]
sample_labels = Y_test[:5]
print("\nSample Inputs (0-4):\n", sample_inputs)
print("\nSample Labels (0-4):\n", sample_labels)

# Get activations
# The activations will be a 2D array where each row corresponds to a sample and each column corresponds to a hidden neuron
activations = activation_model.predict(sample_inputs)

print("\nSample Activations:\n", activations)


print("\nEnter Figure 2 . . .")
# Plot activations for each sample
# Each row corresponds to a sample, each column to a hidden neuron
# The plot shows how each hidden neuron responds to the input samples
# The x-axis represents the hidden neurons, and the y-axis represents the activation values
# Activation Value: Output of a neuron after it applied the activation function
# Higher values indicate stronger activation, which means the neuron is more "active" for that input
plt.figure(figsize=(8, 4))
for i in range(len(sample_inputs)):
    plt.plot(activations[i], label=f"Sample {i} (Label: {sample_labels[i]})", marker='o')

plt.title("Hidden Layer Activations for Test Samples")
plt.xlabel("Neuron Index")
plt.ylabel("Activation Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
