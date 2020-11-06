import random


class Function:
    a = random.uniform(-1.2, 1.2)
    b = random.uniform(-0.4, 0.4)

    @classmethod
    def f(cls, x):
        return cls.a * x + cls.b
