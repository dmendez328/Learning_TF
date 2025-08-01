import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Bidirectional LSTM reads sequences forward and backward, capturing more context
# return_sequences=True is used to allow stacking another LSTM on top
# Final Dense layer uses sigmoid for binary classification
# Dataset is padded to ensure all sequences are the same length

# Load IMDB data (binary sentiment classification)
vocab_size = 10000
max_len = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Bidiretional wrapper lets a recurrent layer (like LSTM, GRU, SimpleRNN) process input sequences in both forward and backward directions
# In a standard LSTM: each time step only gets past context
# In a bidirectional LSTM: each timestep gets past + future context

# Model definition
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(x_train, y_train, validation_split=0.2, epochs=3, batch_size=64)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")