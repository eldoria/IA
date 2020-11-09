import numpy as np


def pooling(img_np, type_pooling):
    result_np = np.zeros((img_np.shape[0] // 2, img_np.shape[1] // 2), dtype=np.int)
    l1 = 0
    c1 = 0
    for line in range(0, img_np.shape[0], 2):
        for col in range(0, img_np.shape[1], 2):
            patch = img_np[line:line + 2, col:col + 2]
            if type_pooling == "max":
                result_np[c1][l1] = return_max(patch)
                l1 += 1
            elif type_pooling == "mean":
                result_np[c1][l1] = return_mean(patch)
                l1 += 1
        c1 += 1
        l1 = 0
    return result_np


def return_max(array):
    result = 0
    for line in range(array.shape[0]):
        for col in range(array.shape[1]):
            if array[line][col] > result:
                result = array[line][col]
    return result


def return_mean(array):
    result = 0
    for line in range(array.shape[0]):
        for col in range(array.shape[1]):
            result += array[line][col]
    result /= (array.shape[0] + array.shape[1])
    return result


if __name__ == "__main__":
    test_tab = np.array([
        [0, 2, 3, 2, 4, 5],
        [4, 1, 8, 4, 3, 6],
        [0, 2, 5, 3, 5, 9],
        [4, 3, 4, 8, 9, 0],
        [7, 6, 6, 6, 9, 1],
        [8, 9, 1, 4, 1, 2],
    ], dtype=np.int)
    test = pooling(test_tab, "max")
    print(test)
