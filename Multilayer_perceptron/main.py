from neural_network import NeuralNetwork
import random


class Data:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets


training_data = [
    Data([1, 0], [1]),
    Data([0, 1], [1]),
    Data([1, 1], [0]),
    Data([0, 0], [0])
]

if __name__ == '__main__':

    neural_network = NeuralNetwork(2, 3, 1)

    for i in range(10000):
        # data = random.choice(training_data)
        for data in training_data:
            neural_network.train(data.inputs, data.targets)

    for data in training_data:
        print(neural_network.feedforward(data.inputs))

