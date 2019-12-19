import numpy as np
from core.activations import relu, sigmoid, softmax, tanh
import types

ACTIVATIONS = {
    "relu": relu,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "tanh": tanh
}


class Perceptron(object):

    """
    Simple perceptron implementation


    Parameters
    ----------
    inputs : array_like
        an array of input values

    bias : float
        the perceptron bias

    weights : array_like
        an array of synaptic weights

    activation: str || function
        the activation function that will be applied to the output.
        Can be a predefined function name or a custom function that receive one parameter

    """

    def __init__(self, inputs, bias, weights, activation):
        if len(inputs) != len(weights):
            raise Exception('Inputs and weights must have same length')
        else:
            self.inputs = inputs
            self.bias = bias
            self.weights = weights

        if type(activation) is str:
            if activation in ACTIVATIONS.keys():
                self.activation = ACTIVATIONS[activation]
            else:
                raise Exception("Function doesn't exists")

        elif type(activation) is types.FunctionType:
            self.activation = activation

    def _combine(self):
        return self.bias + np.sum(np.multiply(self.weights, self.inputs))

    def activate(self):
        return self.activation(self._combine())
