import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

'''
Step 1: Load and inspect data

Load the Iris dataset, set it equal to X
X is the features, Y is the labels
'''

# The Iris dataset is a classic dataset for classification tasks
# The Iris dataset contains 150 samples of iris flowers with 4 features each (sepal length, sepal width, petal length, petal width)

iris = load_iris()
X = iris.data  # Shape: (150, 4)
# Reshape labels to be a 2D array (150, 1) for consistency
Y = iris.target.reshape(-1, 1)  # Shape: (150, 1)
print("\nIris Dataset:", iris)
print("\nX:", X)
print("\nY:", Y)

print("\nFeatures shape:", X.shape)
print("\nLabels shape:", Y.shape)
print("\nUnique classes:", np.unique(Y))

'''
Step 2: Preprocess - Normalize & One-hot encode


'''

# Normalize features (standardization: zero mean, unit variance)
# StandardScaler() recenters the data so the avg is 0 and the std is 1
# fit_transform() computes the mean and stddev and applies the transformation 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled Features:\n", X_scaled)

# One-hot encode labels
# OneHotEncoder converts categorical labels into a binary matrix
# sparse_output=False returns a dense array instead of a sparse matrix
# encoder.fit_transform() fits the encoder and transforms the labels
encoder = OneHotEncoder(sparse_output=False)
Y_onehot = encoder.fit_transform(Y)
print("\nOne-hot Encoded Labels:\n", Y_onehot)

# Train-test split
# train_test_split() splits the dataset into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing
# random_state=42 ensures reproducibility, kind of like a seed for random number generation
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_onehot, test_size=0.2, random_state=42)

'''
Step 3: Build model
'''

# The shape of the input layer is (4,) since we have 4 features
# The hidden layer has 16 neurons with ReLU activation
# The output layer has 3 neurons (one for each class) with softmax activation for multi-class classification
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 output classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

'''
Step 4: Train model
'''

history = model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test))

'''
Step 5: Plot loss and accuracy
'''

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
