# 🧠 Learn TensorFlow through 40 Projects

Welcome to the **TensorFlow Learning Series** — a hands-on project-based curriculum designed to help you master TensorFlow, neural networks, and deep learning fundamentals, one project at a time.

This repository contains **40 structured projects** across **6 progressive phases**, each focusing on practical applications and deepening your understanding of key machine learning and deep learning concepts using TensorFlow.

---

## 📚 Project Breakdown

### 🔰 PHASE 1: Fundamentals of TensorFlow & ML (Projects 1–10)
Get started with TensorFlow syntax, loss functions, optimizers, data pipelines, and visualization tools.

| # | Project | Focus |
|--:|:--------|:------|
| 1 | Hello TensorFlow | `tf.constant`, `tf.Variable`, `tf.function`, graph execution |
| 2 | Linear Regression from Scratch | `tf.GradientTape`, custom training loop |
| 3 | Logistic Regression | Classification, sigmoid, binary cross-entropy |
| 4 | Multi-Class Classifier on Iris Dataset | Softmax, categorical cross-entropy, Keras Sequential API |
| 5 | Loss Functions & Optimizers Playground | MSE, MAE, categorical_crossentropy, Adam, SGD |
| 6 | Data Pipeline Basics | `tf.data.Dataset`, batching, shuffling |
| 7 | Model Evaluation & Overfitting | Validation split, EarlyStopping, underfitting vs. overfitting |
| 8 | Custom Layers & Models | Subclassing `tf.keras.layers.Layer` & `tf.keras.Model` |
| 9 | Save & Load Models | `model.save()`, `model.load_weights()`, SavedModel format |
|10 | TensorBoard Visualizations | `tf.summary`, graph training/validation metrics |

---

### 📈 PHASE 2: Neural Network Design & Real Data (Projects 11–20)
Build and tune dense neural networks on structured and image data.

| # | Project | Focus |
|--:|:--------|:------|
|11 | Deep Feedforward Network on MNIST | Dense layers, ReLU, dropout |
|12 | Hyperparameter Tuning | KerasTuner, batch size, learning rate |
|13 | Fashion MNIST Classifier | Preprocessing, data augmentation |
|14 | Feature Engineering with Tabular Data | StandardScaler, embedding categorical variables |
|15 | Regression with Housing Data | Real-world regression task |
|16 | Custom Training Loop + Metrics | `tf.keras.metrics`, `GradientTape` |
|17 | Weight Initialization Strategies | He/Xavier initialization, training effects |
|18 | Implementing Batch Normalization | Why and when to use it |
|19 | Multi-Input Models | Combine text + numeric inputs |
|20 | Model Deployment with TF Serving | Export, test locally with Docker |

---

### 🧠 PHASE 3: CNNs for Vision (Projects 21–26)
Learn how to build CNNs and process images for classification and localization.

| # | Project | Focus |
|--:|:--------|:------|
|21 | Build a CNN for CIFAR-10 | `Conv2D`, `MaxPooling`, `Flatten`, `Dense` |
|22 | Image Augmentation & Preprocessing | `tf.image`, `ImageDataGenerator` |
|23 | Transfer Learning with MobileNet | Feature extraction, fine-tuning |
|24 | Residual Connections (ResNet Blocks) | Custom ResNet-style CNN |
|25 | Object Localization (Bounding Boxes) | Multi-output regression with ConvNets |
|26 | Build an Image Classifier App | Export model, GUI or web-based inference |

---

### ⏳ PHASE 4: RNNs, LSTMs & Sequence Modeling (Projects 27–32)
Learn how to process sequential data like text and time series.

| # | Project | Focus |
|--:|:--------|:------|
|27 | Basic RNN for Character Generation | `SimpleRNN`, one-hot encoding |
|28 | Sentiment Analysis on IMDB | Embedding, LSTM, text preprocessing |
|29 | Time Series Forecasting (Univariate) | Sliding window, walk-forward validation |
|30 | Stock Price Prediction with LSTM | Multi-step forecasting, `return_sequences` |
|31 | Bidirectional & Stacked LSTMs | Sequence-to-sequence understanding |
|32 | Sequence Classification with Attention | Attention mechanism on input sequences |

---

### 💡 PHASE 5: Autoencoders, GANs & Embeddings (Projects 33–36)
Dive into generative models, embeddings, and unsupervised learning.

| # | Project | Focus |
|--:|:--------|:------|
|33 | Denoising Autoencoder | Clean image reconstruction |
|34 | Variational Autoencoder (VAE) | Latent space sampling, probabilistic modeling |
|35 | GAN for Fashion MNIST | Generator, Discriminator, adversarial loss |
|36 | Word Embeddings from Scratch | Word2Vec-style skip-gram model |

---

### 🔬 PHASE 6: Advanced Topics & Research (Projects 37–40)
Explore cutting-edge research tools and replicate advanced architectures.

| # | Project | Focus |
|--:|:--------|:------|
|37 | Multi-Modal Deep Learning | Combine image and text inputs |
|38 | Transformer for Text Classification | Implement small Transformer from scratch |
|39 | BERT Fine-Tuning | HuggingFace + TF for NLP classification |
|40 | Research Replication Project | Reimplement recent deep learning paper in TensorFlow |

---

## 🧰 Neural Network Design Cheatsheet (Key Concepts)

**Activation Functions**: `ReLU`, `tanh`, `sigmoid`, `softmax`, `swish`  
**Common Layers**: `Dense`, `Conv2D`, `LSTM`, `Embedding`, `BatchNormalization`, `Dropout`, `Flatten`  
**Initializers**: `he_uniform`, `glorot_uniform`  
**Optimizers**: `Adam`, `SGD`, `RMSProp`, `Nadam`  
**Losses**: `MSE`, `binary_crossentropy`, `categorical_crossentropy`  
**Metrics**: `accuracy`, `precision`, `recall`, `AUC`  
**TF APIs**: Sequential, Functional, Subclassing  
**Tools**: `tf.data`, TensorBoard, SavedModel, `@tf.function`

---

## 🚀 Getting Started

### Clone the Repository
```bash
git clone https://github.com/yourusername/tensorflow-projects.git
cd tensorflow-projects
