import random


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
        self.limit_learning_rate = 0.01

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

        # w0 * x + w1 * y + w2 * 1(bias = 1) = 0
        # w1 * y = - w0 * x - w2
        # y = -(w0/w1) * x - (w2/w1)
        return -(w0/w1) * x - (w2/w1)

    def adapt_learning_rate(self):
        print("The learning rate is : " + str(self.learning_rate))
        # Decrease the learning rate in order to be more and more precise and if the perceptron can't find the solution
        # because the learning rate is too low, reset it
        if self.learning_rate > self.limit_learning_rate:
            self.learning_rate *= 0.75
        else:
            print("The algorithm take too long to determine a solution, learning_rate is reset")
            self.learning_rate = 5
            # Allow the program to be more an more precise between each reset in order to prevent cases where
            # the algorithm can't converge because the learning rate is always too high
            self.limit_learning_rate *= 0.5
