import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Before now, we only handled two classes
# We can attempt to handle more than two classes using Softmax, One-Hot encoding, and Categorical cross-entropy loss

# We will generate 3 classes of point, each centered around a different number, and build a softmax classifier to predict which class each x belongs

'''
Step 1: Generate synthetic 1D data for 3 classes

Create classes which are arrays with 100 values each, with a different average and a 0.4 standard deviation
X_train is the classes all in one, stacked vertically on each other
Y_train is what should be output, 100 0's, 1's, and 2's resepectively
You take Y_train, add anotehr dimension, and make it spit out what the one-hot should look like; a 1 wherever the value is, and a 0 in all other spots
'''

# Create 100 samples per class centered at different values
class_0 = tf.random.normal(shape=(100,), mean=2.0, stddev=0.4)
class_1 = tf.random.normal(shape=(100,), mean=5.0, stddev=0.4)
class_2 = tf.random.normal(shape=(100,), mean=8.0, stddev=0.4)
print("\nClass 0:", class_0)
print("\nClass 1:", class_1)
print("\nClass 2:", class_2)

# concat(..., axis=n) -> axis=0 Stack along rows (vertically), asix=1 Stack along columns (horizontally)

# Combine into one dataset
X_train = tf.concat([class_0, class_1, class_2], axis=0)
print("\nx_train:", X_train)

# Create corresponding labels: 0, 1, or 2
Y_train = tf.concat([
    tf.zeros(100, dtype=tf.int32),
    tf.ones(100, dtype=tf.int32),
    tf.ones(100, dtype=tf.int32) * 2
], axis=0)
print("\nY_train:", Y_train)

# One-hot encoding turns categorical class labels (e.g. 0, 1, 2) into vectors where one entry is 1 (hot spot) and all others are 0

# One-hot encode labels: 0 → [1, 0, 0], 1 → [0, 1, 0], 2 → [0, 0, 1]
Y_train_onehot = tf.one_hot(Y_train, depth=3)
print("\nY_train_onehot:", Y_train_onehot)

'''
Step 2: Initialize weights and biases for 3 classes
'''

# W: shape = (1, 3) → 1 feature going to 3 output classes
W = tf.Variable(tf.random.normal(shape=(1, 3)))
b = tf.Variable(tf.random.normal(shape=(3,)))

'''
Step 3: Prediction function (softmax over logits)

Predicting adds another dimension to x, wultiplies x by W, adds b, and then puts it through a softmax activation function
The softmax activation outputs a value in the interval (0, 1), interpreted as a probability, as the sum is 1
'''


def predict(x):

    # Turns a 1D tensor into a 2D column vector so we can do matrix multiplication
    # axis=0 -> Adds a new dimension at the start; axis=1 -> Add a new dimension in the middle, after each element
    x = tf.expand_dims(x, axis=1)  # Shape: (batch_size, 1)
    # Computes the raw, unnormalized outputs ("logits") for each class
    logits = tf.matmul(x, W) + b   # Shape: (batch_size, 3)
    # Converts logits into probabilities using the softmax function
    # Softmax turns raw scores (called ligits) into probabilities between 0 and 1, sum to 1 across all classes
    return tf.nn.softmax(logits, axis=1)

'''
Step 4: Loss function (categorical cross-entropy)
'''

# tf.clip_by_value(tensor, clip_value_min, clip_value_max)
# tensor: The input tensor; clip_value_min: Lower limit; clip_value_max: Upper limit
# Makes values below clip min, clip min, and makes values above clip max, clip max

def compute_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)  # Avoid log(0)
    return -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))

'''
Step 5: Training loop
'''

learning_rate = 0.1
epochs = 300

for epoch in range(epochs):

    with tf.GradientTape() as tape:
        y_pred = predict(X_train)
        loss = compute_loss(Y_train_onehot, y_pred)

    gradients = tape.gradient(loss, [W, b])
    W.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])

    if (epoch + 1) % 50 == 0:
        acc = tf.reduce_mean(
            tf.cast(tf.argmax(y_pred, axis=1) == tf.cast(Y_train, tf.int64), tf.float32)
        )
        print(f"Epoch {epoch+1:3d}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

'''
Step 6: Plot decision boundaries
'''

x_vals = tf.linspace(0.0, 10.0, 200)
y_probs = predict(x_vals)
y_preds = tf.argmax(y_probs, axis=1)

plt.figure(figsize=(8, 4))
plt.scatter(X_train, Y_train, alpha=0.5, label="Training Data")
plt.plot(x_vals, y_preds, color="red", label="Predicted Class", linewidth=2)
plt.title("Softmax Classifier: 3-Class Decision Boundary")
plt.xlabel("x")
plt.ylabel("Class")
plt.legend()
plt.grid(True)
plt.show()



'''
# Trying this on my own

# Generating a bunch of classes
# Create 100 samples per class centered at different values
class_0 = tf.random.normal(shape=(100,), mean=1.0, stddev=0.2)
class_1 = tf.random.normal(shape=(100,), mean=5.0, stddev=0.2)
class_2 = tf.random.normal(shape=(100,), mean=10.0, stddev=0.2)
class_3 = tf.random.normal(shape=(100,), mean=15.0, stddev=0.2)
print("\nClass 0:", class_0)
print("\nClass 1:", class_1)
print("\nClass 2:", class_2)
print("\nClass 3:", class_3)

# Combine into one dataset
X_train = tf.concat([class_0, class_1, class_2, class_3], axis=0)
print("\nx_train:", X_train)

# Create corresponding labels: 0, 1, 2, or 3
Y_train = tf.concat([
    tf.zeros(100, dtype=tf.int32),
    tf.ones(100, dtype=tf.int32),
    tf.ones(100, dtype=tf.int32) * 2,
    tf.ones(100, dtype=tf.int32) * 3
], axis=0)
print("\nY_train:", Y_train)

# One-hot encoding turns categorical class labels (e.g. 0, 1, 2) into vectors where one entry is 1 (hot spot) and all others are 0

# One-hot encode labels: 0 → [1, 0, 0], 1 → [0, 1, 0], 2 → [0, 0, 1]
Y_train_onehot = tf.one_hot(Y_train, depth=4)
print("\nY_train_onehot:", Y_train_onehot)

# W: shape = (1, 4) → 1 feature going to 3 output classes
W = tf.Variable(tf.random.normal(shape=(1, 4)))
b = tf.Variable(tf.random.normal(shape=(4,)))

def predict(x):

    # Turns a 1D tensor into a 2D column vector so we can do matrix multiplication
    # axis=0 -> Adds a new dimension at the start; axis=1 -> Add a new dimension in the middle, after each element
    x = tf.expand_dims(x, axis=1)  # Shape: (batch_size, 1)
    # Computes the raw, unnormalized outputs ("logits") for each class
    logits = tf.matmul(x, W) + b   # Shape: (batch_size, 4)
    # Converts logits into probabilities using the softmax function
    # Softmax turns raw scores (called ligits) into probabilities between 0 and 1, sum to 1 across all classes
    return tf.nn.softmax(logits, axis=1)

# tf.clip_by_value(tensor, clip_value_min, clip_value_max)
# tensor: The input tensor; clip_value_min: Lower limit; clip_value_max: Upper limit
# Makes values below clip min, clip min, and makes values above clip max, clip max

def compute_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)  # Avoid log(0)
    return -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))

learning_rate = 0.1
epochs = 400

for epoch in range(epochs):

    with tf.GradientTape() as tape:
        y_pred = predict(X_train)
        loss = compute_loss(Y_train_onehot, y_pred)

    gradients = tape.gradient(loss, [W, b])
    W.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])

    if (epoch + 1) % 50 == 0:
        acc = tf.reduce_mean(
            tf.cast(tf.argmax(y_pred, axis=1) == tf.cast(Y_train, tf.int64), tf.float32)
        )
        print(f"Epoch {epoch+1:3d}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")


# Note that we get an accuracy of 0.75 with the set up above because of how class 3's mean is 20
# When one class center is much larger than the others, the logits may be extremely large or small causing softmax probabilities to saturate early during trainnig
# With one-hot labels like [0, 0, 0, 1], the model may bias to earlier classes at first and because softmax involves exponential, it may take longer to break out that pattern
# When you use and alter variables of the same name like above, say you uncomment the second hald of this script, then the accuracy of the first part wil lgo down due to data leakage
'''