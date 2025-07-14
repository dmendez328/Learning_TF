import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# This project is about experimenting and developing intuition for how different loss function adn optimizers affect learning behavior
# It is best to try different loss function: MSE, Binatry Cross-Entropy, Categorical Cross-Entropy
# Compare Optimizers: SGD, Adam, RMSProp, ...
# Visualize convergence and training dynamics

'''
Step 1: Create a synthetic binary classification problem

X_train is 200 random values in the interval [0, 10], then it is normalized right after
Y_train is 200 values describing if the respective index in X_train is greater than 5, so it is either 1.0 or 0.0
Reshapes X_train from 1D to 2D
'''

# X ∈ [0, 10], class = 1 if x > 5
X_train = tf.random.uniform(shape=(200,), minval=0, maxval=10)
X_train = (X_train - tf.reduce_mean(X_train)) / tf.math.reduce_std(X_train)
Y_train = tf.cast(X_train > 5, tf.float32)
print("X_train:", X_train)
print("Y_train:", Y_train)

# Reshape input to be 2D: (200, 1)
X_train = tf.expand_dims(X_train, axis=1)
print("X_train Reshaped:", X_train)

'''
Step 2: Choose a loss function and optimizer

mse_loss is the mean squared error loss function, recall that MSE is widely used, especially for regression tasks
binary_crossentropy_loss is the binary cross entropy function
We choose an optimizdr type instead of doing gradient descent manually
'''

# Pick one loss function to experiment with:
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def binary_crossentropy_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))


# Optimizer: An algorithm that adjusts the weights of the model to minimize the loss function
# In previous projects, we manually did gradient descent, but now we can use built in ones

# SGD -> Doesn't adapt, it's fixed for all weights; best for simple problems; can get stuck in local minima, can oscillate if noisy data
# RMSprop -> Keeps a moving average of squared gradients, divides the gradient by the root of the average, helps normalize the learning rate for each weight; Best for RNNs; sensitive to hyperparameters
# Adam -> Combines momentum and RMSprop; best for most deep learning tasks, sparse data, and high dimensional problems; Can overfit if not regularized, may not generalize as well in some cases compared to SGD

# Choose optimizer
# Uncomment the one you'd like to use . . .
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

'''
Step 3: Initialize model parameters
'''

W = tf.Variable(tf.random.normal(shape=(1,), stddev=0.05)) # W initialized to a 1D vector
b = tf.Variable(tf.zeros(())) # b initialized to a constant

'''
Step 4: Prediction and training logic

Our predict function uses a sigmoid activation function
'''

# sigmoid outputs in a range of (0, 1)
# sigmoid is for binary classification, probability outputs
def predict(x):
    return tf.sigmoid(tf.matmul(x, tf.reshape(W, (-1, 1))) + b)  # logistic regression

# Track loss over epochs
loss_history = []

epochs = 300

for epoch in range(epochs):

    with tf.GradientTape() as tape:
        y_pred = predict(X_train)
        
        # We can use two different types of loss
        loss = binary_crossentropy_loss(Y_train, y_pred)
        # loss = mse_loss(Y_train, y_pred)

    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    loss_history.append(loss.numpy())

    if (epoch + 1) % 40 == 0:
        preds = tf.cast(predict(X_train) > 0.5, tf.float32)
        acc = tf.reduce_mean(tf.cast(preds == Y_train, tf.float32))
        print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

# Note that many times you get an accuracy of 1.o, which is never ok because no model can ever really be 100% accurate
# The fact we have no noise contributes to the 100% accuracy
# In real-worl datasets with noise or overlap, 1.0 accuracy is suspicious
# IF you see 1.0 accuracy on training and validation - might be overfitting
# Note that you should normalize input: when using a gradient-based optimizer, before feeding input to mose NN, dense or fully connected layers
# You don't have to normalize for tree-based models (not deep learning), using pretrained networks, for categorical / discrete inputs

'''
Step 5: Plot training loss
'''
'''
plt.figure(figsize=(8, 4))
plt.plot(loss_history, label="Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()
'''