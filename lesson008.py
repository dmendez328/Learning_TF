import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
Step 1: Generate synthetic data for 3 classes
'''

# 3 class clusters, 200 points each
class_0 = tf.random.normal((200,), mean=2.0, stddev=0.4)
class_1 = tf.random.normal((200,), mean=6.0, stddev=0.4)
class_2 = tf.random.normal((200,), mean=10.0, stddev=0.4)

# Features
X = tf.concat([class_0, class_1, class_2], axis=0)
X = tf.expand_dims(X, axis=1)  # Shape: (600, 1)

# Labels (0, 1, 2)
Y = tf.concat([
    tf.zeros(200, dtype=tf.int32),
    tf.ones(200, dtype=tf.int32),
    tf.ones(200, dtype=tf.int32) * 2
], axis=0)

# Shuffle the dataset
indices = tf.random.shuffle(tf.range(len(X)))
X = tf.gather(X, indices)
Y = tf.gather(Y, indices)

'''
Step 2: Create training dataset
'''

BATCH_SIZE = 32
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.shuffle(600).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

'''
Step 3: Build a softmax classifier
'''

# A Dense Layer is used for multi-class classification, it learns how to transform inputs
# Input shape is (1,) since we have one feature
# The output layer should have n many neurons, when your NN classifies n classes 
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes → 3 logits → softmax to get probabilities
])

# Use sparse categorical loss (labels are ints, not one-hot)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''
Step 4: Train the model
'''

history = model.fit(dataset, epochs=30)

'''
Step 5: Visualize class probabilities
'''

# Create 300 evenly spaced values from 0 to 12
x_vis = tf.linspace(0.0, 12.0, 300)
x_vis = tf.reshape(x_vis, (-1, 1))

# Predict class probabilities
probs = model.predict(x_vis)

plt.figure(figsize=(10, 5))
for i in range(3):
    plt.plot(x_vis, probs[:, i], label=f"Class {i} Probability")

plt.title("Softmax Output Across Input Space")
plt.xlabel("Input x")
plt.ylabel("Class Probability")
plt.legend()
plt.grid(True)
plt.show()
