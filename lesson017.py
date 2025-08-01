import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal, HeNormal, GlorotUniform
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# This script demonstrates how to compare different weight initializers in a neural network model
# It uses the MNIST dataset to train models with various initializers and compares their performance

# There are 3 main types of initializers: RandomNormal, HeNormal, and GlorotUniform
# RandomNormal initializes weights with a normal distribution
# HeNormal is designed for layers with ReLU activation
# GlorotUniform (also known as Xavier) is designed for layers with sigmoid or tanh activation

# Initializers can significantly affect model convergence and performance
# Use training curves to visualize differences in convergence speed and final accuracy

# Load and preprocess the MNIST dataset (digits 0–9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print("\nFirst 5 x_train:\n", x_train[:5])
print("\nFirst 5 y_train:\n", y_train[:5])

# Function to build a model with a given initializer
def build_model(initializer):
    model = Sequential([
        Dense(128, activation='relu', kernel_initializer=initializer, input_shape=(784,)),
        Dense(64, activation='relu', kernel_initializer=initializer),
        Dense(10, activation='softmax', kernel_initializer=initializer)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define different initializers to compare
initializers = {
    "RandomNormal": RandomNormal(mean=0.0, stddev=0.05),
    "HeNormal": HeNormal(),
    "GlorotUniform": GlorotUniform()
}

# Dictionary to store training histories for each initializer
histories = {}

# Train a model with each initializer
for name, initializer in initializers.items():
    print(f"\nTraining with initializer: {name}")
    model = build_model(initializer)
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=0)
    histories[name] = history

# Plot training & validation accuracy for comparison
plt.figure(figsize=(10, 6))
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f"{name} Val Acc")
plt.title("Validation Accuracy by Weight Initializer")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
