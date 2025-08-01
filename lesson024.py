import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Input, Add, Activation, BatchNormalization, Flatten, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Converts the datatype of the array; converting pixel values into floats
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Converts class labels (integers) into one-hot encoded vectors
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)
print("\ny_train (FIrst 5):", y_train[:5])

# Residual Blocks: Building block used in deep NNs that helps the model learn deeper representations without degradation
# Degradation Problem: As you make a NN deeper, you might expect it to perform better, but it might make training harder, accuracey may get worse, gradients vanish or explode
# A residual block lets the network skip one or more layers using a shortcut connection
# Basic Idea: Instead of learning the output H(x) directly, the network learns the residual F(x) = H(x) - x, so the final output becomes F(x) + x
# Instead of learning H(x) -> the full mapping from input x to desired output, residual learning says, learn the change from input -> output
# Even if F(x) is small or close to 0, x is preserved; If the optimal mapping is just to pass input through unchanged, the block can easily learn F(x) = 0, so output = x; 
# x -> The input to the residual block; F(x) -> Output from the main path; H(x) - > The final output of the clock (what the network maps x to)

# You know to use a residual block when you're building a very deep NN, and you want to solve or avoid these problems
# As you stack more layers, performance gets worse, not better, not because of overfitting but because the model fails to learn anything useful in very deep paths
# Redisual blocks help when . . . Training Loss Stalls -> Gradients flow more easily through skip paths; Model underperforms deeper -> Skip connections allow identity mappings if needed; Network is > 10-20 layers -> Residuals preserve low-level features + stability


# Define a basic residual block
def residual_block(x, filters):
    shortcut = x  # Save the input for the skip connection

    # First conv layer
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second conv layer
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    # Add the shortcut to the output
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Build the mini-ResNet model
inputs = Input(shape=(32, 32, 3)) # 32 x 32 RGB images, 3 color channels
x = Conv2D(32, (3,3), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Block 1
x = residual_block(x, 32)
x = MaxPooling2D()(x) # Downsamples the feature map by 2x

# Block 2
x = residual_block(x, 32) 
x = MaxPooling2D()(x)  # Downsamples the feature map by 2x

x = Conv2D(64, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile and train
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=64,
    verbose=2
)

# Evaluate and visualize
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Mini ResNet Accuracy on CIFAR-10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()