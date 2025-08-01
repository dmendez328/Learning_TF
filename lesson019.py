import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, Concatenate, Normalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

# This script demonstrates how to build a multi-input model in TensorFlow
# It combines text data and numeric features to predict a binary outcome

# Simulated dataset: each sample has a short text and 2 numeric features
# Expresses short review, giving sentiment
texts = [
    "good product", "bad quality", "excellent item", "poor condition",
    "works well", "terrible experience", "highly recommend", "not worth it"
]

# Corresponding labels for the texts (1 = positive, 0 = negative) and numeric features
# Each text corresponds to a pair of numeric features
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# Numeric features for each text
# Each row corresponds to a text, with 2 numeric features
numeric_data = np.array([
    [4.5, 100], [2.0, 250], [5.0, 80], [1.5, 300],
    [4.0, 90], [1.0, 400], [5.0, 75], [2.5, 220]
], dtype="float32")
print("\nNumeric Data:\n", numeric_data)

# Tokenizing is the process of making raw text into smaller units (tokens) that a model can understand
# It converts text into sequences of integers where each integer represents a word
# NNs can't directly process raw strings, tokenization turns this into numerical data
# Tokenize and pad text
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_texts = pad_sequences(sequences, maxlen=4)

# Normalize numeric features
normalizer = Normalization() # Computes the mean and variance from the dataset, used to scale input
normalizer.adapt(numeric_data) # Computes the stats over numeric_data

# Convert labels to NumPy array
labels = np.array(labels)
print("\nFirst 5 labels:\n", labels[:5], "\nlabels shape:", labels.shape)

# Define model inputs
# This model takes two inputs: text and numeric data
# The text input is a sequence of integers (tokenized words), and the numeric input is an array of numeric features
text_input = Input(shape=(4,), name="text_input") # input for text
num_input = Input(shape=(2,), name="numeric_input") # input for numeric data

# Text branch
# Embedding: Learns a dense vector representation (size 8) for each word index
x_text = Embedding(input_dim=100, output_dim=8)(text_input) # embed words
# GlobalAveragePooling1D: Averages the embeddings across the sequence length
x_text = GlobalAveragePooling1D()(x_text) # average embedding

# Numeric branch
x_num = normalizer(num_input) # normalize numeric input

# Merge both branches
x = Concatenate()([x_text, x_num])
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Build and compile model
model = Model(inputs=[text_input, num_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    {"text_input": padded_texts, "numeric_input": numeric_data},
    labels,
    epochs=15,
    batch_size=2,
    verbose=0,
    validation_split=0.25
)

# Plot training history
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Multi-Input Model Accuracy")
plt.grid(True)
plt.show()
