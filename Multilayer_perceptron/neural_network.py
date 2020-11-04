from matrix import Matrix
import math as mt


def sigmoid(x):
    return 1 / (1 + mt.exp(-x))


def derivative_sigmoid(y):
    # sigmoid(x)' = sigmoid(x) * (1 - sigmoid(x))
    return y * (1 - y)


def pow(x):
    return x * x


class NeuralNetwork:
    def __init__(self, nb_input, nb_hidden, nb_output):
        self.nb_input = nb_input
        self.nb_hidden = nb_hidden
        self.nb_output = nb_output

        self.weights_ih = Matrix(nb_hidden, nb_input).randomize()
        self.weights_ho = Matrix(nb_output, nb_hidden).randomize()

        self.bias_h = Matrix(nb_hidden, 1).randomize()
        self.bias_o = Matrix(nb_output, 1).randomize()

        self.learning_rate = 0.1

    def feedforward(self, inputs_array):
        # Generate hidden outputs
        inputs = Matrix.from_array(inputs_array)
        hidden = Matrix.multiply_m(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden.map(sigmoid)

        # Generate output's output
        output = Matrix.multiply_m(self.weights_ho, hidden)
        output.add(self.bias_o)
        output.map(sigmoid)


        return output.to_array()

    def train(self, inputs_array, targets_array):
        inputs = Matrix.from_array(inputs_array)
        targets = Matrix.from_array(targets_array)

        # Generate hidden outputs
        hidden = Matrix.multiply_m(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden.map(sigmoid)

        # Generate output's output
        outputs = Matrix.multiply_m(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)

        # Calculate output error
        output_error = Matrix.substract(targets, outputs)

        # calculate output gradient
        gradient = Matrix.map_(outputs, derivative_sigmoid)
        gradient = gradient.multiply_h(output_error)
        gradient.multiply_h(self.learning_rate)

        # Calculate output deltas
        hidden_t = Matrix.transpose(hidden)
        weights_ho_deltas = Matrix.multiply_m(gradient, hidden_t)
        self.weights_ho.add(weights_ho_deltas)
        self.bias_o.add(gradient)

        # Determine hidden error
        weights_ho_t = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.multiply_m(weights_ho_t, output_error)

        # Calculate hidden gradient
        hidden_gradient = Matrix.map(hidden, derivative_sigmoid)
        hidden_gradient = hidden_gradient.multiply_h(hidden_errors)
        hidden_gradient.multiply_h(self.learning_rate)

        # Calculate hidden deltas
        inputs_t = Matrix.transpose(inputs)
        weights_ih_deltas = Matrix.multiply_m(hidden_gradient, inputs_t)
        self.weights_ih.add(weights_ih_deltas)
        self.bias_h.add(hidden_gradient)

