from core.perceptron import Perceptron
import numpy as np

inputs = [-0.8, 0.2, -0.4]
weights = [1.0, -0.75, 0.25]


def power2(X):
    # Custom function
    return np.power(X, 2)


def main():
    # Testing Perceptron

    # Neuron with custom activation function
    neuron1 = Perceptron(inputs, -0.5, weights, power2)
    # Neuron with predefined activation function
    neuron2 = Perceptron(inputs, -0.5, weights, 'tanh')

    print('Neuron 1 ->', neuron1.activate())
    print('Neuron 2 ->', neuron2.activate())


if __name__ == '__main__':
    main()
