import numpy as np
from core.nodes.perceptron import Perceptron
from core.layers.perceptron import PerceptronLayer
from core.layers.input import InputLayer


def sigmoid_Derivative(x):
    return x * (1 - x)


inputs = np.array([[0, 0, 1],
                   [1, 1, 1],
                   [1, 0, 1],
                   [0, 1, 1]])

x1 = [0, 0, 1]
x2 = [1, 1, 1]
x3 = [1, 0, 1]
x4 = [0, 1, 1]

outputs = np.array([[0, 1, 1, 0]]).T


def main():

    input_layer = InputLayer()
    input_layer.addNode(x1)
    input_layer.addNode(x2)
    input_layer.addNode(x3)
    input_layer.addNode(x4)

    layer = PerceptronLayer(num_nodes=4, num_inputs=len(x1),
                            activation='sigmoid')

    for iteration in range(20):
        results = layer.fit(input_layer.getValues())
        print('Epoch {}'.format(iteration + 1))
        print(results)

        adjustments = []

        for res in results:
            error = outputs - res
            adjustments.append(error * sigmoid_Derivative(res))

        for i, a in zip(input_layer.getValues(), adjustments):
            layer.backpropagate(np.dot(np.array(i).T, a))


if __name__ == '__main__':
    main()
