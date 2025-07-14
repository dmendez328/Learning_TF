import tensorflow as tf
import numpy as np

# Build a logistic regression model with high level syntax

'''
Step 1: Generate synthetic classification dataset

X is 1000 random points in the [0, 10] interval
Y is a binary output of the expression X > 5
X is then reshaped to be a vector
'''

# Create 1000 points between 0 and 10
X = tf.random.uniform(shape=(1000,), minval=0, maxval=10)

# Binary label: 1 if x > 5, else 0
Y = tf.cast(X > 5, tf.float32)

# Reshape X to (1000, 1) to treat as feature vector
X = tf.expand_dims(X, axis=1)

print("\nSample features and labels:")
print("\nX[0:5]:", X[:5])
print("\nY[0:5]:", Y[:5])

'''
Step 2: Create tf.data.Dataset

Creates a TF dataset object that is used to deed data into a model during training
'''

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
# Each element of the dataset is now a tuple: (feature, label)
# It pairs up data so each example in x is matched with the corresponding label in Y

'''
Step 3: Shuffle, batch, and preprocess

Shuffles, batches, and prefetches the dataset to improve both training randomness and execution performance
'''

BATCH_SIZE = 32

# Shuffle the dataset, then batch it
# buffer_sie=1000 means it will maintain a buffer of 1000 elements and randomly sample from it
# dataset.shuffle(buffersize=...) helps prevent overfitting and improves model generalization
# .batch(BATCH_SIZE) groups data into batches of the specified size, instead of doing one sample at a time, the model processes that amount of samples per timestep
# prefetch() prepares the next batch while the model trains on the current one
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Note that you should shuffle before batching

'''
Step 4: Normalize features using map()

The function normalizes the values, relative to the batch
'''

def normalize_batch(x, y):
    x_mean = tf.reduce_mean(x, axis=0)
    x_std = tf.math.reduce_std(x, axis=0)
    x_norm = (x - x_mean) / x_std
    return x_norm, y

# .map() transforms each element (e.g. each (x, y) pair) in your dataset by passing it through a function you define
dataset = dataset.map(normalize_batch)
print("\nDataset (After Map):", dataset)

'''
Step 5: Iterate over the dataset

Prints whats in the first batch's x and y batches
'''

# dataset.take() grabs just the first batch from the dataset 
print("\nIterating through 1 batch:")
for batch_x, batch_y in dataset.take(1):
    print("\nX (batch):", batch_x.numpy()[:5])
    print("\nY (batch):", batch_y.numpy()[:5])

'''
Step 6 (Optional): Feed into a simple model
'''

# Creates a simple, feedforward NN with 1 layer, 1 output, and a sigmoid activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(1,))
])

# Compiling the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train using the tf.data dataset
model.fit(dataset, epochs=5)
