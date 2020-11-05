import random


def f(x):
    return 0.5 * x + 0.4


class Point:
    def __init__(self):
        self.x = random.uniform(-1, 1)
        self.y = random.uniform(-1, 1)
        self.bias = 1.0
        if self.y >= f(self.x):
            self.label = 1
        else:
            self.label = -1
