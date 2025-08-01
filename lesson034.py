import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# This script helps understand Variational Autoencoders - encoding data into a distribution and not just a point

# Latent Distribution: Encode inputs as mean and standard deviation
# Reparameterization Trick: Enables backpropogation through sampling
# KL Divergence: Regularizes latent space to follow a unit Gaussian
# Reconstruction Los + KL Loss = Total VAE Loss

# -------------------------------
# 1. Load and Preprocess MNIST
(x_train, _), (x_test, _) = mnist.load_data()
# Normalizing data
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# Flatten images (28x28 → 784)
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# -------------------------------
# 2. Sampling Layer
# Sampling Layer: Custom layer in Variational Autoencoders (VAEs) taking the mean and log variance vectos and outputs a random sample z from the learned distribution N(mu, sima^2)
# Can't just use samples from N(mu, sigma^2) because sampling is non-differentiable -> No Gradient Descent
# A latent vactor is a compressed input, the latent distribution is all the possible latent vectors the model believes could correspond to the input
# But you take the mean and variance and use reparameterization to make z differentiable
class Sampling(layers.Layer):
    def call(self, inputs):
        # z_mean: Mean of the latent distribution for the input sample, predicted by the encoder
        # z_log_var: Log of the variance - used for numerical stability, predicted by the encoder
        z_mean, z_log_var = inputs
        # Sample random noise form a standard normal distribution, epsilon is just random noise from N(0,1) distribution
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        # z = μ + σ * ε (reparameterization trick)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# -------------------------------
# 3. Build the VAE Encoder
latent_dim = 2  # for 2D visualization

# Encoder learns to transform input data to a probabilistic latet representation
encoder_inputs = layers.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(encoder_inputs)
x = layers.Dense(128, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# -------------------------------
# 4. Build the VAE Decoder
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation='relu')(latent_inputs)
x = layers.Dense(256, activation='relu')(x)
decoder_outputs = layers.Dense(784, activation='sigmoid')(x)

decoder = Model(latent_inputs, decoder_outputs, name='decoder')
decoder.summary()

# -------------------------------
# 5. VAE Model (Custom Training Step)
# -------------------------------
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        # Just run inputs through the encoder and decoder
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Safe binary crossentropy handling
            bce = tf.keras.losses.binary_crossentropy(data, reconstruction)
            if len(bce.shape) == 1:
                reconstruction_loss = tf.reduce_mean(bce)
            else:
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(bce, axis=1))

            # KL Divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }

# Instantiate and compile
vae = VAE(encoder, decoder)
vae.compile(optimizer='adam', loss=None)

# -------------------------------
# 6. Train the VAE
# -------------------------------
vae.fit(x_train, epochs=30, batch_size=128)

# -------------------------------
# 7. Visualize the Latent Space
# -------------------------------
z_mean, _, _ = encoder.predict(x_test)

plt.figure(figsize=(6, 6))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c='gray', alpha=0.5, s=1)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.title("2D Latent Space of Test Set")
plt.grid(True)
plt.show()

# -------------------------------
# 8. Generate Digits from Latent Space
# -------------------------------
n = 15  # grid size
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Sample points on a grid in latent space
grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[
            i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size
        ] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.axis('off')
plt.title("Generated Digits from Latent Grid")
plt.show()