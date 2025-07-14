import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Logistic regression is a method used to model the probability of a binary outcome based on one or more predictor variables
# Out Classification: If x > 5, label = 1; Else, 0

'''
Step 1: Generate binary classification data

X_train is a bunch of random points on the interval (0, 10)
Y_train spits out a 0.0 or 1.0 depending on if the X_train value > 5

Then create an array of values from 0 through 99, shuffle it, then depending on first 10 numbers, those numbers are the indeces in Y_train to be flipped to add noise
'''

# Create 100 points between 0 and 10
X_train = tf.random.uniform(shape=(100,), minval=0, maxval=10)

# Create binary labels: 1 if x > 5, else 0
Y_train = tf.cast(X_train > 5, tf.float32) # Makes the list into 0.0's and 1.0's
print("\nY_train:", Y_train)

# Add noise by flipping some labels
all_indeces = tf.range(100) # Create a tensor with values [0, ..., 99]
print("\nAll Indeces:", all_indeces)
shuffled_indeces = tf.random.shuffle(all_indeces) # Shuffle the indeces randomly
print("\nShuffled Indeces:", shuffled_indeces)
flip_indeces = shuffled_indeces[:10] # Select the first 10 indeces (random and unique)
print("\nFlipped Indeces:", flip_indeces)

# tf.gather(Y_train, flip_indeces) -> Get current label values
# 1 - (...) -> Flip labels
# tf.reshape(flip_indeces, (-1, 1)) -> Format for scatter update
# tensor_scatter_nd_update(...) -> Apply those changes to Y_train

# Take Y_train, and for the flip_indeces, replace the current value with its opposite
Y_train = tf.tensor_scatter_nd_update(Y_train, tf.reshape(flip_indeces, (-1, 1)), 1 - tf.gather(Y_train, flip_indeces))
# Note that the -1 in the shape spot tells TF, "Figure out this dimension based on the total number of elements"
print("\nY_train:", Y_train)



'''
Step 2: Init model parameters

Init the weight and bias
'''

W = tf.Variable(tf.random.normal(shape=[]))
b = tf.Variable(tf.random.normal(shape=[]))

'''
Step 3: Define prediction function (sigmoid)

We establish the predeiction function which is a neuron taking input x, multiplying it by W, adding b, and putting the output through a sigmoid activation function
'''

def predict(x):
    logits = W * x + b
    return tf.sigmoid(logits) # Squash between 0 and 1

'''
Step 4: Define binary cross-entropy loss

Set up loss function which is Binary Cross-Entropy, used in binary clasigication
'''

def compute_loss(y_true, y_pred):
    # Avoid log(0) by clipping, but we need to put a max value too, because that's what the function requires
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

'''
Step 5: Training loop with gradient descent
'''

learning_rate = 0.05
epochs = 300

for epoch in range(epochs):

    with tf.GradientTape() as tape:
        y_pred = predict(X_train)
        loss = compute_loss(Y_train, y_pred)

    gradients = tape.gradient(loss, [W, b])
    W.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}: Loss = {loss:.4f}, W = {W.numpy():.4f}, b = {b.numpy():.4f}")

'''
Step 6: Plot the sigmoid curve with decision boundary
'''

x_vals = tf.linspace(0.0, 10.0, 100)
y_probs = predict(x_vals)

plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_probs, label="Sigmoid Curve", color="red")
plt.scatter(X_train, Y_train, alpha=0.5, label="Training Data")
plt.axhline(0.5, color='gray', linestyle='--', label="Decision Threshold")
plt.axvline((0.5 - b.numpy()) / W.numpy(), color='green', linestyle='--', label="Learned Boundary")
plt.title("Logistic Regression: Binary Classification")
plt.xlabel("x")
plt.ylabel("Predicted Probability")
plt.legend()
plt.grid(True)
plt.show()


# sigmoid(x) maps real values to [0, 1] - perfect for probability
# binary cross-entropy penalizes confident wrong predictions
# Label noise makes classification more realistic
# Decision boundary -> Learned threshold where output crosses 0.5