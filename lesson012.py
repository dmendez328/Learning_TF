import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# This script demonstrates how to handle overfitting in a simple binary classification task using TensorFlow and Keras.

# Training loss is the loss calculated on the training data, tells how well the model is fitting the data it already has seen
# You want the training loss to decrease steadily because it means the model is learning the patterns in the training set
# Validation loss is calculated on a separate validation set that the model never sees during training, tells how well the model is generalizing to new, unseen data
# Want this to go down with training, but not diverge from training loss too much

# Good learning -> Training loss decreases, validation loss decreases, and they stay close together
# Overfitting (Model is memorizing the training set) -> Training loss decreases, validation loss starts to increase after a point, diverging from training loss
# Underfitting (Model is too simple or not learning) -> Training loss and validation loss are both high, model is not learning enough from the training data

# To reduce training loss, you can:
# 1. Increase model complexity, try different architectures (more layers, more neurons)
# 2. Train for more epochs, but be careful of overfitting though
# 3. Use early stopping to halt training when validation loss stops improving
# 4. Use a better optimizer or adjust learning rate

# To reduce validation loss, you can:
# 1. Use regularization techniques (like dropout or L2 regularization) to prevent overfitting
# 2. Use early stopping to halt training when validation loss stops improving
# 3. Increase the amount of training data if possible, more data can help the model generalize better
# 4. Apply data preprocessing techniques (like normalization or augmentation) to improve model performance

'''
Step 1: Create noisy synthetic binary classification data
'''

np.random.seed(42)
X = np.random.uniform(0, 10, size=(500, 1))
Y = (X[:, 0] > 5).astype(np.float32)
print("\nX:", X[:5], "\nY:", Y[:5])

# Add random noise to make classification harder
flip_mask = np.random.rand(500) < 0.2
Y[flip_mask] = 1 - Y[flip_mask]
print("\nNoisy Y:", Y[:5])

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

'''
Step 2: Define model that easily overfits
'''

# Dropout Layer: Regularization technique that randomly drops (sets to zero) a fraction of neurons during each training step
# L2 Regularization: Adds a penalty term to the loss function that discourages large weight values
# Large weights can cause a model to be too sentsitive to noise, memorize training data (overfit)
# L2 regularization encourages smaller weights, smooths the decision boundary, help prevent overfitting

# use_dropout: Is a dropout layer used? (Dropout Layer helps prevent overfitting)
# use_l2: Is L2 regularization used? (L2 Regularization helps prevent overfitting)
# Function to build a simple model with optional dropout and L2 regularization
def build_model(use_dropout=False, use_l2=False):

    # List to hold layers
    layers = []

    # Regularizer, if use_l2=True, adds a L2 penalty term to the loss
    regularizer = tf.keras.regularizers.l2(0.01) if use_l2 else None

    layers.append(tf.keras.layers.Dense(64, activation='relu', input_shape=(1,), kernel_regularizer=regularizer))
    
    if use_dropout:
        layers.append(tf.keras.layers.Dropout(0.3))

    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

    model = tf.keras.Sequential(layers)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

'''
Step 3: Train base model (no regularization)
'''

# Regularization: Is a set of techniques to constrain or penalize a model to make it simpler and more generalizable
# This model is built without any regularization techniques
base_model = build_model()
history = base_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, verbose=0)

'''
Step 4: Plot overfitting symptoms
'''

# This plots the training loss at each epoch and the validation loss at each epoch
# x axis is the epochs; y axis is the loss value
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Overfitting Example: Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

'''
Step 5: Try Dropout and L2 Regularization
'''

# Sparse: Model has many weights that are zero, many connections between neurons are not used, leading to simpler models and faster inference
# Smooth: Model has small weights, leading to smoother decision boundaries, less sensitive to noise in the data

# L1 Regularization: Encourages sparsity (many weights become zero), adds a penalty proportional to the absolute value of the weights
# L2 Regularization: Encourages smaller weights without forcing them to zero, adds a penalty proportional to the square of weights
# L1 & L2 Regularization: Combines both, useful when you want sparse + smooth weights

# Build a model with dropout and L2 regularization
reg_model = build_model(use_dropout=True, use_l2=True)
history_reg = reg_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, verbose=0)

plt.plot(history_reg.history['loss'], label='Train Loss (Reg)')
plt.plot(history_reg.history['val_loss'], label='Val Loss (Reg)')
plt.title("Regularization: Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

'''
Step 6: Early Stopping (stops training when val_loss stops improving)
'''

early_model = build_model(use_dropout=True, use_l2=True)
early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history_early = early_model.fit(X_train, Y_train,
                                validation_data=(X_test, Y_test),
                                epochs=100,
                                callbacks=[early_stop],
                                verbose=0)

# Plot the training and validation loss for the early stopping model
plt.plot(history_early.history['loss'], label='Train Loss (Early Stop)')
plt.plot(history_early.history['val_loss'], label='Val Loss (Early Stop)')
plt.title("Early Stopping")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
