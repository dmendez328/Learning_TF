import tensorflow as tf
import numpy as np

# This script is supposed to help demonstrate how autoencoders work by building a Denoising Autoencoder
# Denoising Autoencoder: Used to reconstruct clean images from noisy ones

# Autoencoder: Type of NN designed to learn a compressed representation (encoding) of input data and then reconstruct the original input from that encoding
# Autoencoders follow the flow of . . . Input -> Encoder -> Bottleneck -> Decoder -> Output
# Encoder: Compresses the input into a lower-dimensional space (code or latent vector); built with Dense, Conv2D, or LSTM layers
# Bottleneck / Latent Space: This is the compressed representation of the input; forces the network to prioritize and extract key features; the bridge between encoder and decoder
# Decoder: Takes compressed code and attempts to reconstruct the original input; mirrors the encoder, often with symmetric layers

# Autoencoders are best suited for learning compressed, meaningful representations of input data without labels

# Sinusoidal Positional Encoding Function
# Goal: Return a tensor of shape (1, seq_len, d_model) that encodes position information using sine and cosine function
def get_positional_encoding(seq_len, d_model):
    # seq_len: Number of time steps (e.g. 50 for a sequence of 50 tokens)
    # d_model: Dimension of the model's embedding space
    # Creates a matrix of shape (seq_len, d_model) where each value is a scaled version of the position index
    angles = np.arange(seq_len)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model)
    pos_encoding = np.zeros_like(angles)
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2]) # Even indices -> apply sin
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2]) # Odd indices -> apply cos

    # Adds a batch dimension
    # Casts to float32 TF tensor so it can be added to token embeddings during training
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

# Scaled Dot-Product Attention Scores
# Implements scaled dot-product attention
def scaled_dot_product_attention(q, k, v, mask=None):

    # Computes raw attention scores between all pairs of query and key positions
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # Scale by sqrt of key dimension, to prevent vary large values in the dot products
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
 
    # Any position where the mask is 1 gets a huge negative value added so it becomes ~0 after the softmax
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # large negative = masked out

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # Now all scores become probabilities (summing to 1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

# Multi-Head Attention Layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0  # Ensure even split
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Dense layers to project inputs into q, k, v
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        # Output linear layer
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq, depth)

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        attn_output, _ = scaled_dot_product_attention(q, k, v, mask)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        concat_attn = tf.reshape(attn_output, (batch_size, -1, self.num_heads * self.depth))

        return self.dense(concat_attn)

# Transformer Encoder Block
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # Position-wise FFN
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, mask=None, training=False):
        # Multi-head attention + residual + norm
        attn_output = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))

        # Feedforward network + residual + norm
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2

# Sample Usage
# Define model hyperparameters
seq_len = 10
d_model = 64
num_heads = 4
dff = 128

# Random dummy input (batch_size=2, seq_len=10, d_model=64)
x = tf.random.uniform((2, seq_len, d_model))

# Add positional encoding to input
pos_encoding = get_positional_encoding(seq_len, d_model)
x += pos_encoding

# Create encoder layer
encoder_layer = TransformerEncoderLayer(d_model, num_heads, dff)

# Forward pass
output = encoder_layer(x, training=True)

print("\nTransformer encoder output shape:", output.shape)