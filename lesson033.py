import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# This script shows how to train an autoencoder that learns to reconstruct clean images from noisy ones using the MNIST dataset

# --------------------------------------
# 1. Load and Prepare MNIST Data
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values to [0,1] and flatten (28x28 → 784)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# --------------------------------------
# 2. Add Gaussian Noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip values to [0, 1] so pixels remain valid, within normalized range
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# --------------------------------------
# 3. Define the Autoencoder Architecture
input_img = Input(shape=(784,))

# Encoder: compress input
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)  # Bottleneck

# Decoder: reconstruct image
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)  # Output layer with pixel range [0,1]

# Autoencoder model: maps noisy input → clean output
autoencoder = Model(input_img, decoded)

# --------------------------------------
# 4. Compile and Train the Model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(
    x_train_noisy, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# --------------------------------------
# 5. Visualize Denoising Results
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10  # number of examples to display
plt.figure(figsize=(20, 6))

for i in range(n):
    # Noisy input
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    # Denoised output
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

    # Original clean image
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

plt.tight_layout()
plt.show()
