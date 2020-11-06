import Point
import Perceptron
import Function
import matplotlib.pyplot as plt


def show_graph():
    y1 = Function.Function.f(-1)
    y2 = Function.Function.f(1)
    y3 = perceptron.guess_y(-1)
    y4 = perceptron.guess_y(1)
    plt.plot([-1, 1], [y1, y2], 'k--', lw=3)
    plt.plot([-1, 1], [y3, y4], 'r', lw=2)

    plt.axis([-1, 1, -1, 1])
    for pt in points:
        if perceptron.guess([pt.x, pt.y, pt.bias]) == pt.label:
            plt.scatter(pt.x, pt.y, c="green", marker='o', s=30)
        else:
            plt.scatter(pt.x, pt.y, c="red", marker='o', s=55)
    plt.show()


if __name__ == "__main__":
    print("The number of points is randomly generated, there are "
          + str(Point.Point.SIZE_POPULATION) + " points.")
    print("The function is randomly generated and its equation is "
          + str(Function.Function.a) + "x " + str(Function.Function.b))
    points = [0] * Point.Point.SIZE_POPULATION

    for i in range(Point.Point.SIZE_POPULATION):
        points[i] = Point.Point()

    perceptron = Perceptron.Perceptron()

    training_is_over = 0

    while training_is_over == 0:
        training_is_over = 1
        for point in points:
            if perceptron.guess([point.x, point.y, point.bias]) != point.label:
                perceptron.adapt_learning_rate()
                training_is_over = 0
                perceptron.train([point.x, point.y, point.bias], point.label)
        if training_is_over == 1:
            print("The perceptron has found a line that describe pretty well the model")
        else:
            show_graph()
