import random
import Point


def activate(nb):
    if nb >= 0:
        return 1
    else:
        return -1


class Perceptron:
    weights = [0.0] * 3

    def __init__(self):
        self.weights[0] = random.uniform(-1, 1)
        self.weights[1] = random.uniform(-1, 1)
        self.weights[2] = random.uniform(-.5, .5)
        self.learning_rate = 5
        self.limit_learning_rate = 0.001

    def guess(self, inputs):
        the_sum = 0
        for i in range(len(self.weights)):
            the_sum += self.weights[i] * inputs[i]
        return activate(the_sum)

    def train(self, inputs, target):
        guess = self.guess(inputs)
        error = target - guess
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.learning_rate

    def guess_y(self, x):
        w0 = self.weights[0]
        w1 = self.weights[1]
        w2 = self.weights[2]

        return -(w2/w1) - (w0/w1) * x

    def adapt_learning_rate(self):
        print("The learning rate is : " + str(self.learning_rate))
        # If the perceptron can't converge, reset the learning rate
        if self.learning_rate > self.limit_learning_rate:
            self.learning_rate *= 0.6
        else:
            self.learning_rate = 5
