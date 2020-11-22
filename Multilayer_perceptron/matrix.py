import random


class Matrix:
    def __init__(self, nb_rows, nb_columns, nb_dims):
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.nb_dims = nb_dims

        self.data = [[[0 for i in range(nb_columns)] for j in range(nb_rows)] for k in range(nb_dims)]

    def randomize(self):
        # Fill the matrix with random values from -1 to 1
        for i in range(self.nb_rows):
            for j in range(self.nb_columns):
                for k in range(self.nb_dims):
                    self.data[i][j][k] = random.random() * 2 - 1
        return self

    @staticmethod
    def multiply_m(a, b):
        # matrix product
        if a.nb_columns != b.nb_rows:
            print("It's impossible to multiply the matrices because the number of lines of the first matrix is not "
                  "equal to the number of columns of the second matrix")
            return None

        result = Matrix(a.nb_rows, b.nb_columns)

        for i in range(result.nb_rows):
            for j in range(result.nb_columns):
                sum = 0
                for k in range(a.nb_columns):
                    sum += a.data[i][k] * b.data[k][j]
                result.data[i][j] = sum
        return result

    def multiply_h(self, n):
        # hadamard product
        if isinstance(n, Matrix):
            result = Matrix(self.nb_rows, n.nb_columns)
            for i in range(result.nb_rows):
                for j in range(result.nb_columns):
                    sum = 0
                    for k in range(self.nb_columns):
                        sum += self.data[i][k] * n.data[k][j]
                    result.data[i][j] = sum
            return result
        # Scalar product
        else:
            for i in range(self.nb_rows):
                for j in range(self.nb_columns):
                    self.data[i][j] *= n

    def print_matrix(self):
        print(self.data)

    @staticmethod
    def from_array(arr):
        matrix = Matrix(len(arr), 1)
        for i in range(len(arr)):
            matrix.data[i][0] = arr[i]
        return matrix

    def to_array(self):
        arr = []
        for i in range(self.nb_rows):
            for j in range(self.nb_columns):
                arr.append(self.data[i][j])
        return arr

    @staticmethod
    def map_(matrix, func):
        # Apply a function to each element of the matrix
        for i in range(matrix.nb_rows):
            for j in range(matrix.nb_columns):
                matrix.data[i][j] = func(matrix.data[i][j])
        return matrix

    def map(self, func):
        # Apply a function to each element of the matrix
        for i in range(self.nb_rows):
            for j in range(self.nb_columns):
                self.data[i][j] = func(self.data[i][j])
        return self

    def add(self, val):
        # Add a value to each element of the matrix
        if isinstance(val, Matrix):
            for i in range(self.nb_rows):
                for j in range(self.nb_columns):
                    self.data[i][j] += val.data[i][j]
        else:
            for i in range(self.nb_rows):
                for j in range(self.nb_columns):
                    print("AYA")
                    print(self.data[i][j])
                    print(val)
                    self.data[i][j] += val
            return self.data

    @staticmethod
    def substract(a, b):
        result = Matrix(a.nb_rows, a.nb_columns)
        for i in range(result.nb_rows):
            for j in range(result.nb_columns):
                result.data[i][j] = a.data[i][j] - b.data[i][j]
        return result

    @staticmethod
    def transpose(matrix):
        result = Matrix(matrix.nb_columns, matrix.nb_rows)
        for i in range(matrix.nb_rows):
            for j in range(matrix.nb_columns):
                result.data[j][i] = matrix.data[i][j]
        return result
