import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# This script demonstrates how to build a simple softmax classifier using TensorFlow and Keras
# It generates synthetic spiral data, builds a model, trains it, and visualizes the results

# Step 1: Generate synthetic spiral data
# points_per_class: Points in each spiral
# num_classes: Number of spiral classes
def generate_spiral_data(points_per_class=100, num_classes=3):
    # To store the 2D coordinates
    X = []
    # Will store labels for each point
    y = []
    # Generate points for each class
    for class_number in range(num_classes):
        r = np.linspace(0.0, 1, points_per_class)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2  # angle
        X.extend(np.c_[r * np.sin(t), r * np.cos(t)])
        y.extend([class_number] * points_per_class)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# Generate the spiral data
X, y = generate_spiral_data()
print("\nSpiral Data Generated:")
print("\ny:", y)
print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# Visualize the data, putting it into a scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=20)
plt.title("Spiral Data (3 Classes)")
plt.show()

# Step 2: One-hot encode labels
y_onehot = tf.one_hot(y, depth=3)
print("\nOne-hot Encoded Labels:", y_onehot)

# Step 3: Build a simple softmax classifier
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(X, y_onehot, epochs=100, batch_size=32, verbose=0)

# Step 6: Evaluate and visualize
loss, acc = model.evaluate(X, y_onehot, verbose=0)
print(f"\nFinal Accuracy: {acc:.4f}")

# Plot the decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = model.predict(grid, verbose=0)
    Z = np.argmax(pred, axis=1).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='brg')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(model, X, y)
