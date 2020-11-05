import Point
import Perceptron
import matplotlib.pyplot as plt

SIZE_POPULATION = 100


def show_graph():
    y1 = Point.f(-1)
    y2 = Point.f(1)
    y3 = perceptron.guess_y(-1)
    y4 = perceptron.guess_y(1)
    plt.plot([-1, 1], [y1, y2], 'k--', lw=3)
    plt.plot([-1, 1], [y3, y4], 'r', lw=2)

    plt.axis([-1, 1, -1, 1])
    for pt in points:
        if perceptron.guess([pt.x, pt.y, pt.bias]) == pt.label:
            plt.scatter(pt.x, pt.y, c="green", marker='o', s=30)
        else:
            plt.scatter(pt.x, pt.y, c="red", marker='o', s=50)
    plt.show()


if __name__ == "__main__":
    points = [0] * SIZE_POPULATION

    for i in range(SIZE_POPULATION):
        points[i] = Point.Point()

    perceptron = Perceptron.Perceptron()

    training_is_over = 0

    while training_is_over == 0:
        # print(perceptron.weights[0], perceptron.weights[1], perceptron.weights[2])
        training_is_over = 1
        for point in points:
            if perceptron.guess([point.x, point.y, point.bias]) != point.label:
                training_is_over = 0
                perceptron.train([point.x, point.y, point.bias], point.label)
        show_graph()
