# =====================================================================
# COMPLETE BEGINNER-FRIENDLY DEEP LEARNING CONCEPTS IN ONE FILE
# =====================================================================
#
# This script explains the following concepts directly in comments:
#
# 1. What is a Perceptron?
# 2. What are Weights and Bias?
# 3. What are Layers in a Neural Network?
# 4. Forward Propagation
# 5. Backpropagation
# 6. Activation Functions:
#       - Sigmoid
#       - ReLU
#       - Tanh
# 7. Loss Functions:
#       - Mean Squared Error (MSE)
#       - Binary Cross Entropy
# 8. Optimizers:
#       - Stochastic Gradient Descent (SGD)
#       - Adam
# 9. Transformer Concept:
#       - Self Attention
#       - Query, Key, Value
#
# Everything is implemented using NumPy only.
# =====================================================================

import numpy as np


# =====================================================================
# SECTION 1: PERCEPTRON CONCEPT
# =====================================================================
#
# A Perceptron is the smallest building block of a neural network.
#
# Mathematical Formula:
#
#     z = (w1*x1 + w2*x2 + ... + wn*xn) + bias
#     output = activation(z)
#
# Where:
#   x  = input features
#   w  = weights (importance of each input)
#   bias = additional adjustable parameter
#   activation = function that introduces non-linearity
#
# Without activation, neural networks behave like linear regression.
# =====================================================================


# =====================================================================
# SECTION 2: ACTIVATION FUNCTIONS
# =====================================================================
#
# Activation functions introduce NON-LINEARITY.
# Without non-linearity, deep networks collapse into linear models.
# =====================================================================

def sigmoid(x):
    """
    Sigmoid activation function

    Formula:
        1 / (1 + e^(-x))
        

    Output range: (0, 1)
    Used for:
        - Binary classification
        - Output layer probabilities
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid
    Needed for backpropagation

    If s = sigmoid(x)
    derivative = s * (1 - s)
    """
    return x * (1 - x)


def relu(x):
    """
    ReLU (Rectified Linear Unit)

    Formula:
        max(0, x)

    Advantages:
        - Fast
        - Reduces vanishing gradient problem
        - Most commonly used in hidden layers
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU
        1 if x > 0
        0 if x <= 0
    """
    return (x > 0).astype(float)


def tanh(x):
    """
    Hyperbolic Tangent activation

    Formula:
        (e^x - e^-x) / (e^x + e^-x)

    Output range: (-1, 1)

    Better than sigmoid in hidden layers because
    outputs are zero-centered.
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Derivative of tanh
        1 - tanh^2(x)
    """
    return 1 - np.tanh(x)**2


# =====================================================================
# SECTION 3: LOSS FUNCTIONS
# =====================================================================
#
# Loss functions measure how wrong the model predictions are.
# The optimizer tries to minimize this value.
# =====================================================================

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error (MSE)

    Used for regression.

    Formula:
        (1/n) * Σ(y_true - y_pred)^2
    """
    return np.mean((y_true - y_pred)**2)


def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy Loss

    Used for binary classification.

    Formula:
        -[y log(p) + (1-y) log(1-p)]

    Lower value = better model
    """
    epsilon = 1e-9  # prevents log(0)
    return -np.mean(
        y_true*np.log(y_pred+epsilon) +
        (1-y_true)*np.log(1-y_pred+epsilon)
    )


# =====================================================================
# SECTION 4: SIMPLE NEURAL NETWORK ARCHITECTURE
# =====================================================================
#
# We build:
#
# Input Layer (2 neurons)
# Hidden Layer (4 neurons)
# Output Layer (1 neuron)
#
# Layers consist of neurons.
# Each connection between neurons has a weight.
# Each neuron has a bias.
# =====================================================================

np.random.seed(42)

# OR gate dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [1]])

input_size = 2
hidden_size = 4
output_size = 1

# Initialize weights randomly
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


# =====================================================================
# SECTION 5: FORWARD PROPAGATION
# =====================================================================
#
# Forward propagation steps:
#
# 1. Multiply inputs with weights
# 2. Add bias
# 3. Apply activation
# 4. Repeat for next layer
#
# Data flows:
#     Input → Hidden → Output

# =====================================================================
#
# Forward propagation means:
#   Input → Hidden Layer → Output Layer
#
# Mathematical flow:
#
#   z1 = X·W1 + b1
#   a1 = activation(z1)
#
#   z2 = a1·W2 + b2
#   y_pred = activation(z2)
#
# Where:
#   X  = Input
#   W  = Weights
#   b  = Bias
#   z  = Linear combination
#   a  = Activated output
# ==========================================================

def forward_propagation(X, W1, b1, W2, b2):
    """
    Performs forward pass through a 2-layer neural network.
    """

    # Hidden layer linear step
    z1 = np.dot(X, W1) + b1

    # Apply activation (ReLU for hidden layer)
    a1 = relu(z1)

    # Output layer linear step
    z2 = np.dot(a1, W2) + b2

    # Apply activation (Sigmoid for binary classification)
    y_pred = sigmoid(z2)

    # Return all intermediate values (needed for backpropagation)
    return z1, a1, z2, y_pred
# =====================================================================


# =====================================================================
# SECTION 6: BACKPROPAGATION
# =====================================================================
#
# Backpropagation calculates gradients (derivatives)
# and updates weights to reduce loss.
#
# Uses chain rule from calculus.
#
# Steps:
# 1. Compute error at output
# 2. Propagate error backward
# 3. Update weights
# =====================================================================

#
# Backpropagation computes gradients and updates weights.
#
# Core idea:
#   Find how much each weight contributed to the error.
#
# Chain Rule:
#
#   dLoss/dW = dLoss/dOutput * dOutput/dZ * dZ/dW
#
# Steps:
#   1. Compute output error
#   2. Propagate error backward
#   3. Compute gradients
#   4. Update weights
# ==========================================================

def backpropagation(X, y, z1, a1, z2, y_pred, W2, learning_rate):
    """
    Performs backward pass and updates weights.
    """

    m = X.shape[0]  # number of samples

    # -------------------------------
    # STEP 1: Output layer gradient
    # -------------------------------
    #
    # For Binary Cross Entropy + Sigmoid,
    # derivative simplifies to:
    #
    #   dz2 = y_pred - y
    #
    dz2 = y_pred - y

    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # -------------------------------
    # STEP 2: Hidden layer gradient
    # -------------------------------
    #
    # Backpropagate error to hidden layer
    #
    dz1 = np.dot(dz2, W2.T) * relu_derivative(z1)

    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # -------------------------------
    # STEP 3: Update weights (SGD)
    # -------------------------------
    #
    # weight = weight - learning_rate * gradient
    #
    W2_updated = W2 - learning_rate * dW2
    W1_updated = W1 - learning_rate * dW1

    return W1_updated, W2_updated, db1, db2

# ==========================================================

# =====================================================================
# SECTION 7: OPTIMIZER - SGD
# =====================================================================
#
# SGD (Stochastic Gradient Descent):
#
#     weight = weight - learning_rate * gradient
#
# Simple but powerful.
# =====================================================================

learning_rate = 0.1
epochs = 5000

print("Training using SGD...\n")

for epoch in range(epochs):

    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    loss = binary_cross_entropy(y, y_pred)

    # Backprop
    dz2 = y_pred - y
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * relu_derivative(z1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # SGD update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal Predictions (SGD):")
print((y_pred > 0.5).astype(int))


# =====================================================================
# SECTION 8: ADAM OPTIMIZER
# =====================================================================
#
# Adam combines:
#   - Momentum
#   - RMSProp
#
# Uses:
#   m = moving average of gradients
#   v = moving average of squared gradients
#
# Provides adaptive learning rates.
# =====================================================================


# =====================================================================
# SECTION 9: TRANSFORMER CONCEPT (SELF ATTENTION)
# =====================================================================
#
# Transformer does NOT use RNN or CNN.
# It relies entirely on SELF-ATTENTION.
#
# Core idea:
#
# Every word looks at every other word and decides
# how important they are.
#
# Steps:
#   1. Create Query (Q), Key (K), Value (V)
#   2. Compute attention score:
#          Q * K^T
#   3. Scale by sqrt(d_k)
#   4. Apply Softmax
#   5. Multiply by V
# =====================================================================

print("\nTransformer Self-Attention Demo\n")

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Example sequence: 3 tokens, embedding size 4
X_seq = np.random.rand(3, 4)

# Weight matrices
Wq = np.random.rand(4,4)
Wk = np.random.rand(4,4)
Wv = np.random.rand(4,4)

Q = np.dot(X_seq, Wq)
K = np.dot(X_seq, Wk)
V = np.dot(X_seq, Wv)

scores = np.dot(Q, K.T) / np.sqrt(K.shape[1])
attention_weights = softmax(scores)
output = np.dot(attention_weights, V)

print("Attention Weights:\n", attention_weights)
print("\nTransformer Output Representation:\n", output)


# =====================================================================
# END OF COMPLETE DEEP LEARNING CONCEPT FILE
# =====================================================================
