import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# This script demonstrates how to compare different weight initializers in a neural network model
# It uses the MNIST dataset similar to 17

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print("\nFirst 5 x_train:\n", x_train[:5], "\nx_train shape:", x_train.shape)
print("\nFirst 5 y_train:\n", y_train[:5], "\ny_train shape:", y_train.shape)

# Batch normalization is a technique to normalize the inputs of each layer (e.g. Dense -> BatchNorm -> Activation)
# It helps stabilize learning and can lead to faster convergence
# After normalization, the mean is ~ 0 and standard deviation is ~ 1
# It helps speed up convergence, stabilize training, reduce sensitivity to initialization, and acts as a regularizer
# It is typically applied after the Dense layer and before the activation function

# Use batch norm when . . .
# 1. You have deep networks (more than 2-3 layers), helps with vanishing and exploding gradients
# 2. No dropout or regularization, acts like a mild regularizer
# 3. Training is unstable or slow, helps smooth out the optimization landscape 
# 4. You want to use higher learning rates, allows for larger steps in optimization

# You might not need batch norm when . . .
# 1. You have very shallow networks (1-2 layers), normalization may not be necessary
# 2. You are using small datasets, may not help much
# 3. You are using architectures that already include normalization (e.g. ResNet, Inception)

# Best practice is to use it . . .
# 1. After each hidden Dense or Conv layer
# 2. Before the activation function (except for the output layer)

# Model WITHOUT Batch Normalization
def build_plain_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Model WITH Batch Normalization
def build_batchnorm_model():
    model = Sequential([
        Dense(128, input_shape=(784,)),
        BatchNormalization(),            # normalize after Dense, before activation
        Activation('relu'),              # keep activation separate for better control
        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dense(10, activation='softmax')  # output layer stays the same
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and compare models
plain_model = build_plain_model()
bn_model = build_batchnorm_model()

print("\nTraining model WITHOUT BatchNorm:")
plain_history = plain_model.fit(x_train, y_train, validation_split=0.2, epochs=15, batch_size=128, verbose=0)

print("\nTraining model WITH BatchNorm:")
bn_history = bn_model.fit(x_train, y_train, validation_split=0.2, epochs=15, batch_size=128, verbose=0)

# Plot validation accuracy for comparison
plt.figure(figsize=(10, 6))
plt.plot(plain_history.history['val_accuracy'], label='No BatchNorm')
plt.plot(bn_history.history['val_accuracy'], label='With BatchNorm')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
