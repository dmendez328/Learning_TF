import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess CIFAR-10 dataset
# 10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
# Images are 32x32 RGB
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

print("\nFirst 5 training samples:\n", x_train[:5], y_train[:5])


# CNNs are a class of deep learning models that specialize in processing grid-like data, such as images
# They use convolution layers to extract  freatures (edges, textures, shapes)
# CNNs are translation invariant, meaning they recognize patters regardless of location

# Use CNNs whem . . .
# Your input has spatial structure (2D or 3D images)
# You want automatic feature extraction rather than hand-designed filters
# You're working with computer vision problems like classification, segmentation, or object detection

# CNNs are useful for 1D data too, like when there's a local dependency on time or position (e.g. adjacent inputs influence each other) or you want to extract patters across windows in a times-series or sequence

# You can also use CNNs for non-images, like . . .
# Structured 2D data with local relationships
# Text and NLP, CNNs can slide a 1D window across sequences of word embeddings, often used in sentiment analysis or text classification
# Audio and signals, audio signals can be treated as 1D (raw waveform) or 2D (spectrogram)
# Genomics / DNA sequences, CNNs can learn patterns in nucleotide sequences, useful for tasks like gene prediction or disease classification

# Define the CNN architecture
# - 2 Conv layers + MaxPooling
# - Flatten to Dense layers
# - Dropout for regularization

# There are different CNN function for different dimensions: Conv1D, Conv2D, Conv3D

# Pooling layers reduce the spatial dimensions of the feature maps, helping to downsample and reduce computation
# MaxPooling: Downsampling operation that reduces the spatial dimention size of the feature maps while retaining the most important information
# AveragePooling: Averages the values in each region
# GlobalPooling: Averages the entire feature map

# MaxPooling -> Image recognition (keeps strongest features), object detection (preserves the most salient signals), any pattern where peak value matters (e.g. signal spikes, corners); may ignore subtle features or context, very aggressive - throws away a lot
# AveragePooling -> Low noise data (preserves more context than max), feature smoothing (reduce variance / noise), compression and downsampling (don't want harsh filters), signal procesing (smoother representation of inputs); may dilute important features
# GlobalAveragePooling -> Image classification (reduce each channel to a single value), replacing flatten + dense (reduce overfitting and parameter count), mobile models (smaller, more efficient architecture); discards spatial location completely
# GlobalMaxPooling -> Highlighting most activated feature globally, use when presence of a feature matters more than its location

# Pooling (usually MaxPooling) is used after Conv layers to reduce spatial size, retain key info, add translation invariance, and lower parameter count

# Might want to skip pooling if:
# You want high spatial resolution, tasks like semantic segmentation, super resolution, or object localization; pooling can blur or lose fine-grained info
# You stack multiple Convs before pooling, letting early layers extract more complex features before downsampling


model = Sequential([

    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # reduce overfitting
    Dense(10, activation='softmax')  # 10 output classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the CNN
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    verbose=2
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN CIFAR-10 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
