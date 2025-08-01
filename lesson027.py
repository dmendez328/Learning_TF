import tensorflow as tf
import numpy as np
import random

# The goal is to understand how to build a basic character-level RNN
# This teaches tokenizing and preparing character sequences, RNNs for sequence-tosequencing modeling, sampling predictions to generate new text
# Concept: Given a sequence of characters, the RNN learns to predict the next character; you'll feed it text and it will learn the patterns of letter sequences

# Recurrent Neural Network (RNN) is a type of NN designed to process sequential data like time series, audio, text, or anything where order matters
# Unlike traditional feedforward networksm RNNs have loops that allow information to persist
# RNNs process sequences step-by-step memory from previous steps (the hidden state)
# SimpleRNN is a basic version of RNN - later we will use LSTMs and GRUs which are better for long-term dependencies
# The model learns character transitions from the training text
# generate_text() keeps feeding the model its own output to generate sequences

# Load a small corpus
text = "The sun dipped below the horizon, casting golden hues across the sky. Birds chirped in the trees as a warm breeze drifted through the open fields. In the distance, children laughed and chased each other through the tall grass. Everything felt peaceful, as if time had slowed down to admire the beauty of the moment. It was one of those rare evenings where every sound, color, and feeling came together perfectly."
vocab = sorted(set(text)) # Get all unique characters in the text (e.g. letters, punctuation), then sorts those characters so their order is consistent and reproducible
char2idx = {u: i for i, u in enumerate(vocab)} # Creates a dictionary that maps each character u to a unique index i
idx2char = np.array(vocab) # Converts the sorted vocabulary list to a NumPy array
vocab_size = len(vocab) # Simply the number of unique characters (i.e. the size of the vocab), used to define the input size for the embedding layer and define the output size for the dense layer at the end

# Convert text to integers
text_as_int = np.array([char2idx[c] for c in text])

# Sequence length and training examples
seq_length = 20 # Defines the length of each input sequence, the model will look at 20 characters at a time to try to predict the next character.
examples_per_epoch = len(text_as_int) // (seq_length + 1) # Calculates number of non-verlapping sequences of seq_length + 1 you can extract from the full text

# Create training sequences
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# Split input/target
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Shuffle and batch
BATCH_SIZE = 2
BUFFER_SIZE = 100
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Model parameters
embedding_dim = 256
rnn_units = 512

# The Embedding layer is a preprocessing layer that turns interger encoded characters (or words) into dense vectors of real number
# It is stringly recommended in most NLP tasks involving discrete inputs like characters or words

# Build the training model (stateless)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length),
    tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
    tf.keras.layers.Dense(vocab_size)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Train the model
model.fit(dataset, epochs=100)

# Rebuild the model for generation (stateful, batch_size = 1)
generation_model = tf.keras.Sequential([
    tf.keras.Input(batch_shape=(1, None)),
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, stateful=True),
    tf.keras.layers.Dense(vocab_size)
])

# Set weights from trained model
# Copy weights only between matching layers (skip Input layer)
for layer, gen_layer in zip(model.layers, generation_model.layers[1:]):
    if len(layer.get_weights()) == len(gen_layer.get_weights()):
        gen_layer.set_weights(layer.get_weights())

# Generate text function
def generate_text(model, start_string, num_generate=200, temperature=0.9):

    # num_generate: Controls how many characters your model will generate after the initial start_string
    # When you divide the predictions (logits) by temperature, you adjust the probability distribution before sampling the next character
    # Low Temp (< 1.0): More confident, deterministic
    # High Temo (> 1.0): More exploratory, creative
    # Defailt Temp (1.0): Sampling from the model's raw output as-is

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Find and reset any layer that supports reset_states (e.g., RNNs)
    for layer in model.layers:
        if hasattr(layer, "reset_states"):
            layer.reset_states()


    generated = []
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        generated.append(idx2char[predicted_id])

    return start_string + ''.join(generated)

# Try generating text
print("\n--- Generated Text ---")
print(generate_text(generation_model, start_string="The ", num_generate=300))