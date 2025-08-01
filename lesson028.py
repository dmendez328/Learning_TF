import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# This script shows how to build an LSTM based model for binary sentiment classification on text data using the IMDB daraset
# Embedding layer converts word indeces into dense vectors
# LSTM captures word order and sequence dependencies in the review

# Hyperparameters for Natural Language Processing (NLP)
vocab_size = 10000 # Defines max number of unique words (tokens) in vocabulary, the model will only keep top 10000 most frequent words
max_len = 200 # Sets the max length of each input text
embedding_dim = 64 # This is the size of the dense vector that each word will be mapped to in the embedding layer, so each word will be converted into a vector of 64 float values


# The IMDB dataset is for binary sentiment classification
# 50000 total movie reviews from IMDB, 25000 reviews for training and 25000 reviews for testing

# Load IMDB dataset (already preprocessed)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to same length, adjusting for too short of a length
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

# Long Short-Term Memory (LSTM): A special type of recurrent NN to designed remember info over long sequences
# LSTM is a NN architecture that preocesses sequential data, just like a regular RNN, but it fixes the problem of vanishing or exploding gradients
# Each LSTM cell has 3 gates that control the flow of information . . .
# Forget Gate -> Decides what info to throw away from the previous state
# Input Gate -> Decides what new info to store
# Output Gate -> Decides what to output (visible part of the memory)
# The internal memory and gating mechanism make it very good at remembering context over many time steps
# LSTMs are good for NLP, Speech Recognition, Time Series Prediction, Music Generation, DNA Sequence Modeling

# Build model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")