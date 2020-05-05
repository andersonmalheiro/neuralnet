from core.nodes.perceptron import Perceptron


class PerceptronLayer(object):
    def __init__(self, num_nodes, num_inputs, bias=None, activation='linear'):
        self.nodes = []
        if bias:
            for _ in range(num_nodes):
                node = Perceptron(
                    num_inputs, bias=bias, activation=activation)
                self.nodes.append(node)
        else:
            node = Perceptron(
                num_inputs, activation=activation)
            for _ in range(num_nodes):
                self.nodes.append(node)

        self.results = []

    def get_nodes(self):
        return self.nodes

    def fit(self, inputs):
        for node in self.nodes:
            y = node.activate(inputs)
            self.results.append(y)

        return self.results

    def backpropagate(self, weights):
        for node, weight in zip(self.nodes, weights):
            node.update_weights(weight)
