import tensorflow as tf
import numpy as np

# Linear Regression tries to fit a straight line to your data: y = Wx + b
# The goal is to find the best values for W and b that minimize the difference between y' and y


'''
Step 1: Create synthetic training data for y = 3x + 2

X_train will be 100 random points points in the interval [0, 10]
The true slope and true intercept will be 3 and 2, respectively
Y_train is a bunch of points that mimick y = 3x + 2, but it has noise, we will feed this to hopefully get our model to spit out the equation of the line we want it to find which is y = 3x + 2
'''

# Shape (100,) is a Vector (1D) [x1, x2, ...]
# Shape (100, 1) is a column Vector (2D) [[x1], [x2], ...]
# Shape (100, 3) is a Matrix (2D) with 3 features [[f1, f2, f3], ..., [f100, f200, f300]]

# random.uniform(...) Generates values uniformly between a lower and upper bound
# random.normal(...) Generates from a normal (Gaussian) distribution 

# Create 100 random points around x in the element of [0, 10]
X_train = tf.random.uniform(shape=(100,), minval=0, maxval=10)
true_slope = 3.0
true_intercept = 2.0

# Generate y = 3x + 2 + some noise
Y_train = true_slope * X_train + true_intercept + tf.random.normal(shape=(100,), stddev=0.2)

'''
Step 2: Init model parameters

Initialize your weight, W, and your bias, b
'''

# These are the variables we want to learn (initialized randomly)
# We initialize them randomly then let the model adjust them using gradient descent
# Shape [] is 0D, kind of like a constant
W = tf.Variable(tf.random.normal(shape=[]), name="Weight")
b = tf.Variable(tf.random.normal(shape=[]), name="Bias")

'''
Step 3: Define the prediction function

The prediction function is basically establishing a neuron as a function, where Wx + b is the operation the neuron performs
'''

def predict(x):
    return W * x + b

'''
Step 4: Define the loss function (Mean Squared Error)

The loss function computes the loss using MSE, the difference of the true y and predicted y, squared
'''

# MSE = (1/n) SIGMA(y - y')^2
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) # (y - y')^2

'''
Step 5: Training loop using GradientTape

For the amount of epochs we set it to, predict y, compute the loss in comparison to the correct value, then update the weight and bias accordingly
'''

# Set learning rate and number of epochs
learning_rate = 0.01 # A hyperparameter that controls how much the model's wights are adjusted during training
epochs = 200 # The number of times the training data will be fed into it

# A gradient is the direction and magnitude of the steepest increase of a function
for epoch in range(epochs): # Loop epochs number of times

    # The following is a context manager that records all the operations you perform on trainable variables (W or b)
    with tf.GradientTape() as tape:

        y_pred = predict(X_train)
        loss = compute_loss(Y_train, y_pred)

    # Compute gradients with respect to W and b, will tell us how to adjust the model's parameters to reduce loss
    gradients = tape.gradient(loss, [W, b])

    # Manually update the weights using gradient descent
    W.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])

    # Prints every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: Loss = {loss:.4f}, W = {W.numpy():.4f}, b = {b.numpy():.4f}")

'''
Final Results
'''

print("\nTrained Parameters:")
print("Learned W (slope):", W.numpy())
print("Learned b (intercept):", b.numpy())

# Predict on a few test points
x_test = tf.constant([0.0, 1.0, 5.0, 10.0])
y_test_pred = predict(x_test)

print("\nTest Predictions:")
for x, y_pred in zip(x_test.numpy(), y_test_pred.numpy()):
    print(f"x = {x:.1f} → y = {y_pred:.2f}")


# Note that with any noise (high or low), the b will not be exactly 2
# Even with low noise stddev = 0.2, every label y in the dataset is slightly off
# The model learns an approximate best-fit line, not the perfect theoretical one
# If we train for only ~200 epochs, the model might not fully converge to the lowest possible loss
