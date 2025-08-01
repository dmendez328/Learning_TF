# ================================================================
# What is WSL and Why Do I Need It?
# ---------------------------------------------------------------
# WSL (Windows Subsystem for Linux) allows me to run Linux tools 
# inside Windows. It's lightweight, safe, and doesn't replace my OS.
# 
# Docker Desktop for Windows uses WSL 2 under the hood to run
# Linux-based containers. Without WSL, Docker can't run properly.
#
# If I were using a Linux OS already, then I would NOT need WSL.
# ================================================================

# ================================================================
# What is Docker and Why Am I Using It?
# ---------------------------------------------------------------
# Docker lets me package and run apps (like TensorFlow Serving) 
# in containers — self-contained environments that work anywhere.
#
# In this script, I'm using Docker to:
# Load my trained model
# Serve it as a REST API on localhost:8501
# Test it by sending real HTTP prediction requests
#
# This is how machine learning models are deployed in production.
# ================================================================

# ================================================================
# What To Do After Installing Docker Desktop
# ---------------------------------------------------------------
# 1. Make sure Docker is running (open Docker Desktop, check whale icon)
# 2. In a terminal, confirm Docker works:
#       docker --version
#
# 3. Run the container using this command from the project directory:
#       docker run -p 8501:8501 --name=tf_serving_mnist `
#        --mount type=bind,source=${PWD}\saved_models\mnist_model,target=/models/mnist_model `
#        -e MODEL_NAME=mnist_model `
#        -t tensorflow/serving
#
#    OR restart it if it already exists:
#       docker start tf_serving_mnist
#
# 4. Open a second terminal and run the Python client script to test:
#    - This sends a sample input to the server and prints the prediction.
# ================================================================



import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define and train a simple model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)

# ================================
# METHOD 1: Save in .h5 format
# ================================
# Saving in this format saves the entire model architecture, weights, optimizer state, and training configuration
# Works well for smaller models and easy for local deployment and transfer
h5_path = "mnist_model.h5"
model.save(h5_path)
print(f"Model saved in .h5 format at: {h5_path}")

# You can load this model like so:
reloaded_model = load_model(h5_path)
loss, acc = reloaded_model.evaluate(x_test, y_test, verbose=0)
print(f"Reloaded .h5 model test accuracy: {acc:.4f}")

# =========================================
# METHOD 2: Export SavedModel for Serving
# =========================================
# Saves model in TFs SavedModel format, which is a directory with saved_model.pb + variables / and assets / subfolders
# Optimized for production environments, particularly TF serving
export_path = os.path.join("saved_models", "mnist_model", "1")  # versioned path for TF Serving
model.export(export_path)  # Keras 3+ export method
print(f"Model exported for TF Serving at: {export_path}")