import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# This is to see what is available
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Transfer Learning: Reused a pre-trained CNN trained on ImageNet
# Feature Extraction: Used MobileNetV2 to extract powerful image features
# Fine-Tuning: Unfroze the base model to improve performance even more
# Image Resizing:  Adjusted CIFAR-10 images to match MobileNet input size (96x96)

# Transfer learning is fast and effective, especially with limited data
# Pre-trained models (i.e. MobileNet, ResNet, EfficientNet) learn general features
# You can freeze the base model or fine-tune it depending on your task, size, and budget
# Resize input images to match what the base model expects



# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Use only 20,000 training images for speed
x_train, y_train = x_train[:20000], y_train[:20000]
x_test, y_test = x_test[:4000], y_test[:4000]

# Pretrained models require a specific input size, but we use 64 for the MobileNetV2 because this is an intensive program and we need it to run
# MobileNetV2 -> At least 96 96, 3
# ResNet50 -> 224, 224, 3
# EfficientNet80 -> 224, 224, 3
# Resize from (32, 32) → (64, 64) to fit MobileNetV2 input
x_train = tf.image.resize(x_train, [64, 64]) / 255.0
x_test  = tf.image.resize(x_test, [64, 64]) / 255.0
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Load pre-trained MobileNetV2 (without top layer)
# - weights='imagenet': loads trained filters
# - include_top=False: exclude the final classifier
# - input_shape must match resized images
base_model = MobileNetV2(
    input_shape=(64, 64, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers so we only train new head
# .trainable -> Asking if the weights are to be updated during training (False = No, True = Yes)
base_model.trainable = False

# Add a custom classification head on top
inputs = Input(shape=(64, 64, 3))
x = base_model(inputs, training=False) # use base model as a frozen feature extractor
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile and train the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    verbose=2
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy (frozen base): {test_acc:.4f}")
print(f"Test Loss (frozen base): {test_loss:.4f}")


'''
# Optional: Fine-tune base model (unfreeze some layers)

base_model.trainable = True  # unfreeze for fine-tuning

# Recompile with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune entire model
fine_tune_history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64,
    verbose=2
)

# Evaluate after fine-tuning
final_loss, final_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy (after fine-tuning): {final_acc:.4f}")
'''