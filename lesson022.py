import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ImageDataGenerator: Applies transformations to images during training, helps prevent overfitting by creating new ones (modified versions), and optionally handles normalization, rescaling, and validation splits
# rotation_range: Randomly rotates images by up to +/- 15 degrees, helps be rotation invariant
# width_shift_range: Shifts image horizontally by up to 10% of the image width, helps recognize objects off-center
# height_shift_range: Shifts image vertically by up to 10% of the image width, helps recognize objects off-center
# zoom_range: Randomly zooms by up to 10%, helps learn scale-invariant features
# horizontal_flip: Randomly flips images horizontally, for symmetric data like faces or animals
# fill_mode: When shifts/rotations create empty pixels, tells how to fill them
# validation_split: Reserves 20% of the data for validation

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=15, # rotate images randomly (±15 degrees)
    width_shift_range=0.1, # shift image horizontally
    height_shift_range=0.1, # shift image vertically
    zoom_range=0.1, # zoom in
    horizontal_flip=True, # flip horizontally
    fill_mode='nearest', # fill missing pixels
    validation_split=0.2 # split 20% for validation
)

# Create augmented training and validation generators
train_gen = datagen.flow(x_train, y_train, batch_size=64, subset='training')
val_gen = datagen.flow(x_train, y_train, batch_size=64, subset='validation')

# Define the CNN (same as in Project 21)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using the data generator
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    verbose=2
)

# Evaluate on clean test data (not augmented)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy with Augmentation: {test_acc:.4f}")

# Plot training/validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN Accuracy with Data Augmentation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()