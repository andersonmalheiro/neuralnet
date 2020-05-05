import numpy as _np

__all__ = ['sigmoid', 'relu', 'softmax', 'tanh', 'linear']


def sigmoid(X):
    return 1 / (1 + _np.exp(-X))


def relu(X):
    return _np.maximum(0, X)


def softmax(X):
    expo = _np.exp(X)
    expo_sum = _np.sum(_np.exp(X))
    return expo/expo_sum


def tanh(X):
    return _np.tanh(X)


def linear(X):
    return X
