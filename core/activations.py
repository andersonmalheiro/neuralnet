import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def relu(X):
    return np.maximum(0, X)


def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum


def tanh(X):
    return np.tanh(X)
