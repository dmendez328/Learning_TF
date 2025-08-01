import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import random

# Generate synthetic image data with colored squares
# Each image is 64x64 black with one colored square
# X: Images of shape (64, 64, 3), all black except for one colored squre
# y_class: The class of the square (0 = red, 1 = green, 2 = blue)
# y_bbox: The bounding box [x, y, width, height] 
def generate_data(num_samples=2000):
    X = np.zeros((num_samples, 64, 64, 3), dtype=np.float32)
    y_class = np.zeros((num_samples,), dtype=np.int32)
    y_bbox = np.zeros((num_samples, 4), dtype=np.float32)  # [x, y, w, h]

    for i in range(num_samples):
        color_id = random.randint(0, 2)  # red, green, or blue
        x, y = random.randint(5, 44), random.randint(5, 44)
        size = random.randint(10, 20)
        X[i, y:y+size, x:x+size, color_id] = 1.0
        y_class[i] = color_id
        y_bbox[i] = [x / 64, y / 64, size / 64, size / 64]  # normalized

    return X, y_class, y_bbox

# y_class predicts the color of the square ()
# y_bbox are the coordinates
X, y_class, y_bbox = generate_data()
y_class = tf.keras.utils.to_categorical(y_class, 3) # One-hot encoding

# Split dataset into 80% train and 20% validation
split = int(0.8 * len(X))
x_train, x_val = X[:split], X[split:]
y_class_train, y_class_val = y_class[:split], y_class[split:]
y_bbox_train, y_bbox_val = y_bbox[:split], y_bbox[split:]

# A model can make more than one prediction at the same time called a multi-output model
# The nth-output model ends in n many output layers, each has a name, target, and a loss

# Build multi-output model
inputs = Input(shape=(64, 64, 3))
x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = MaxPooling2D()(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)

class_output = Dense(3, activation='softmax', name='class_output')(x) # Color of the box
bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(x) # Predicted box position

model = Model(inputs=inputs, outputs=[class_output, bbox_output])
model.compile(
    optimizer='adam',
    loss={
        'class_output': 'categorical_crossentropy',
        'bbox_output': 'mse'
    },
    metrics={'class_output': 'accuracy'}
)
model.summary()

# Train the model
history = model.fit(
    x_train,
    {'class_output': y_class_train, 'bbox_output': y_bbox_train},
    validation_data=(x_val, {'class_output': y_class_val, 'bbox_output': y_bbox_val}),
    epochs=15,
    batch_size=32,
    verbose=2
)

# True is the true color and Pred is the predicted colore (0 = red, 1 = green, 2 = blue)
# Visualize predictions on validation images
def show_predictions(num=10):
    # Runs the trained model on the first num validation images
    preds = model.predict(x_val[:num]) # Multi-output model -> preds is a list (0 = classification predictions; 1 = Bounding box predictions)
    class_preds = np.argmax(preds[0], axis=1) # Predicted class label (An int: 0, 1, 2)
    bbox_preds = preds[1] # 4 predicted values [x, y, w, h] per image (normalized between 0 and 1)

    # Looping through each image and display 
    for i in range(num):
        img = x_val[i] # Image to display
        label = np.argmax(y_class_val[i]) # True class (e.g. 1 for green)
        pred_class = class_preds[i] # What the model guessed
        pred_box = bbox_preds[i] # Models predicted bounding box

        plt.figure(figsize=(2, 2))
        plt.imshow(img)
        ax = plt.gca()

        # Predicted bounding box (red)
        x, y, w, h = pred_box
        rect = plt.Rectangle((x*64, y*64), w*64, h*64, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        plt.title(f"True: {label}, Pred: {pred_class}")
        plt.axis("off")
        plt.show()

show_predictions()