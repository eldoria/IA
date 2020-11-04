import random
import Point


def activate(nb):
    if nb >= Point.f(nb):
        return 1
    else:
        return -1


class Perceptron:
    weights = [0.0] * 3

    def __init__(self):
        self.weights[0] = random.uniform(-1, 1)
        self.weights[1] = random.uniform(-1, 1)
        self.weights[2] = random.uniform(-1, 1)

    def guess(self, inputs):
        the_sum = 0
        for i in range(len(self.weights)):
            the_sum += self.weights[i] * inputs[i]
        return activate(the_sum)

    def train(self, inputs, target):
        error = target - self.guess(inputs)
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * 0.2

    def guess_y(self, x):
        w0 = self.weights[0]
        w1 = self.weights[1]
        w2 = self.weights[2]

        return -(x * w0 - w2) / w1
