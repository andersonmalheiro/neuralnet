from core.nodes.input import InputNode


class InputLayer(object):
    def __init__(self):
        self._nodes = []

    def addNode(self, value):
        node = InputNode(value)
        self._nodes.append(node)

    def getValues(self):
        return [value.getValue() for value in self._nodes]
