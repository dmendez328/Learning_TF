import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
Step 1: Generate the data
'''

X = tf.random.uniform(shape=(1000,), minval=0, maxval=10)
Y = tf.cast(X > 5, tf.float32)
X = tf.expand_dims(X, axis=1)

# Split: 80% training, 20% validation
split_index = int(0.8 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
Y_train, Y_val = Y[:split_index], Y[split_index:]

'''
Step 2: Create datasets
'''

BATCH_SIZE = 32

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

# Shuffle only training set
train_ds = train_ds.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

'''
Step 3: Define a simple binary classifier
'''

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

'''
Step 4: Add early stopping
'''

# If val_loss doesn't improve for 5 epochs, stop training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

'''
Step 5: Train and validate
'''

# model.fit(...) trains the model using the training dataset
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stopping],
    verbose=1
)

'''
Step 6: Plot training and validation loss
'''

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
