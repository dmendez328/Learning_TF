import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
Step 1: Create nonlinear 2D classification dataset (concentric circles)

Generate points in a circle and label them based on their distance from the center.
'''

# n is the number of points, r is the radius of the inner circle
def generate_circle_data(n=1000, r=5.0):
    # Generate n random points in 2D -> (n, 2)
    x = tf.random.uniform((n, 2), minval=-10.0, maxval=10.0)
    dists = tf.norm(x, axis=1)  # Euclidean distance from origin (i.e. distance formula performed from center)
    y = tf.cast(dists < r, tf.float32)  # Inside = class 1, Outside = class 0
    return x, y

X, Y = generate_circle_data(n=1000, r=5.0)
print("\nX:", X)
print("\nY:", Y)

# Visualize data
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', alpha=0.5)
plt.title("Nonlinear Classification Dataset (Circle)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.show()

'''
Step 2: Build dataset

We will use TensorFlow's Dataset API to create a training dataset.
Use dataset to make X and Y tensors, then shuffle, batch, and prefetch.
'''

BATCH_SIZE = 32
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

'''
Step 3: Build model with a hidden layer

Define a simple neural network with 2 inputs, one (16 neuron) hidden layer with a ReLU activation, and an output layer with a sigmoid activation.
Compile the model with binary crossentropy loss and Adam optimizer.
'''

# Hidden layers add non-linearity and depth to a model
# Hidden layers allow models to learn complex patterns
# ReLU(x) = max(0, x) is a common activation function
# sigmoid(x) = 1 / (1 + exp(-x)) is used for binary classification, outputs probability between 0 and 1 - all add to 1

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),  # Hidden layer with ReLU
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) # metrics tells Keras to track additional information (e.g. accuracy) during training

'''
Step 4: Train the model

Train the model using the dataset created in Step 2, 20 epochs.
'''

model.fit(dataset, epochs=20)

'''
Step 5: Visualize decision boundary

Visualize the decision boundary by creating a grid of points and predicting their class probabilities.
Plot the decision boundary along with the original data points.
'''

# Generate grid of (x1, x2) values
xx, yy = np.meshgrid(np.linspace(-10, 10, 300),
                     np.linspace(-10, 10, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, probs, levels=50, cmap='bwr', alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', edgecolor='k', s=20)
plt.title("Decision Boundary (with Hidden Layer)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.colorbar(label='Predicted Probability')
plt.show()
