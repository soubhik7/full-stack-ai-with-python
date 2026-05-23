"""
deep_learning_models.py

This file demonstrates:

1. Deep Neural Network (DNN) for classification
2. Autoencoder for dimensionality reduction
3. Convolutional Neural Network (CNN) for image classification
4. Recurrent Neural Network (RNN - LSTM) for sequence prediction

All examples use small built-in datasets for easy understanding.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import LSTM, SimpleRNN, Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


# ============================================================
# 1️⃣ DEEP NEURAL NETWORK (DNN)
# ============================================================

def deep_neural_network_example():
    print("\nRunning Deep Neural Network Example...\n")

    # Load MNIST dataset (handwritten digits 0-9)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values (0-255 → 0-1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Flatten 28x28 image into 784 input features
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Create model
    model = Sequential()

    # Input layer + Hidden layer 1
    model.add(Dense(128, activation='relu', input_shape=(784,)))

    # Hidden layer 2
    model.add(Dense(64, activation='relu'))

    # Output layer (10 digits)
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(x_train, y_train, epochs=3, batch_size=32)

    # Evaluate
    loss, accuracy = model.evaluate(x_test, y_test)
    print("DNN Accuracy:", accuracy)


# ============================================================
# 2️⃣ AUTOENCODER
# ============================================================

def autoencoder_example():
    print("\nRunning Autoencoder Example...\n")

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Input layer
    input_img = Input(shape=(784,))

    # Encoder (compress data)
    encoded = Dense(64, activation='relu')(input_img)

    # Decoder (reconstruct data)
    decoded = Dense(784, activation='sigmoid')(encoded)

    # Autoencoder model
    autoencoder = Model(input_img, decoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=3,
                    batch_size=256,
                    shuffle=True)

    # Reconstruct images
    decoded_imgs = autoencoder.predict(x_test[:5])

    # Show original vs reconstructed
    plt.figure(figsize=(10, 4))
    for i in range(5):
        # Original
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

        # Reconstructed
        plt.subplot(2, 5, i + 6)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.show()


# ============================================================
# 3️⃣ CONVOLUTIONAL NEURAL NETWORK (CNN)
# ============================================================

def cnn_example():
    print("\nRunning CNN Example...\n")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Add channel dimension (required for CNN)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = Sequential()

    # Convolution layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

    # Pooling layer
    model.add(MaxPooling2D((2, 2)))

    # Flatten before Dense
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(64, activation='relu'))

    # Output layer
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3, batch_size=32)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("CNN Accuracy:", accuracy)


# ============================================================
# 4️⃣ RECURRENT NEURAL NETWORK (RNN - LSTM)
# ============================================================

def rnn_example():
    print("\nRunning RNN (LSTM) Example...\n")

    # Create simple sequence dataset
    # Example: predict next number in sequence
    X = []
    y = []

    for i in range(100):
        X.append([i, i+1, i+2])
        y.append(i+3)

    X = np.array(X)
    y = np.array(y)

    # Reshape for RNN (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()

    # LSTM layer
    model.add(LSTM(50, activation='relu', input_shape=(3, 1)))

    # Output layer
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=20, verbose=0)

    # Test prediction
    test_input = np.array([[50, 51, 52]])
    test_input = test_input.reshape((1, 3, 1))

    prediction = model.predict(test_input)
    print("Predicted next value:", prediction[0][0])


# ============================================================
# MAIN FUNCTION
# ============================================================

if __name__ == "__main__":
    deep_neural_network_example()
    autoencoder_example()
    cnn_example()
    rnn_example()
