# This lesson was done exclusively in Google Colab

# This script demonstrates how to build a simple image classifier using TensorFlow and Gradio.

'''
Codeblock #1
'''

# Install Gradio in Colab3
!pip install -q gradio

'''
Codeblock #2
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

# The CIFAR-10 dataset contains 60000 color images of suze 32 by 32; 10 classes, 6000 images each
# Classes are: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data() # Load CIFAR-10
x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalize the pixel value
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
print("\ny_train (First 5):", y_train[:5])

# Note that the order of the classes in the list does matter
# Class labels
class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Train a lightweight CNN (or load a pretrained one)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') # Because there are 10 different class names
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train for speed — you can increase epochs later
model.fit(x_train, y_train, epochs=8, batch_size=64, validation_split=0.1, verbose=2)

'''
Codeblock #3
'''

import gradio as gr
from PIL import Image

# Gradio expects a function that takes an image and returns a prediction
def predict_cifar10(img):
    # Resize to 32x32 and normalize
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = img.reshape(1, 32, 32, 3)
    
    # Get model prediction
    pred = model.predict(img)[0]
    confidences = {class_names[i]: float(pred[i]) for i in range(10)}
    return confidences

# Launch the Gradio interface
demo = gr.Interface(
    fn=predict_cifar10,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Image Classifier",
    description="Upload a 32x32 image (or any image) and see what the model predicts!"
)

demo.launch()

'''
Codeblock #4
'''

# To turn off the share 
demo.close()