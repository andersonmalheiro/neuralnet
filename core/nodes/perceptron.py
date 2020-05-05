import numpy as np
import core.activations as af
import types

__all__ = ['Perceptron']

ACTIVATIONS = {
    "relu": af.relu,
    "sigmoid": af.sigmoid,
    "softmax": af.softmax,
    "tanh": af.tanh,
    "linear": af.linear,
}

np.random.seed(1)


class Perceptron(object):

    """
    Simple perceptron implementation


    Parameters
    ----------
    num_inputs : int
        number of inputs

    bias : float
        the perceptron bias

    weights : array_like
        an array of synaptic weights

    activation: str || function
        the activation function that will be applied to the output.
        Can be a predefined function name or a custom function that receive one parameter

    """

    def __init__(self, num_inputs, bias=0, activation='linear'):
        self.bias = bias
        if num_inputs <= 0:
            raise Exception('`num_inputs` must be a positive integer value')
        else:
            self.weights = 2 * np.random.random((3, 1)) - 1

        if type(activation) is str:
            if activation in ACTIVATIONS.keys():
                self.activation = ACTIVATIONS[activation]
            else:
                raise Exception("Function doesn't exists")

        elif type(activation) is types.FunctionType:
            self.activation = activation

    def update_weights(self, w):
        self.weights = w

    def __combine(self, inputs):
        return self.bias + np.dot(inputs, self.weights)

    def activate(self, inputs):
        return self.activation(self.__combine(inputs))
