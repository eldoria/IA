import numpy as np
import random


def randomize_tab(size):
    tab = np.zeros((size, size), dtype=np.int)
    for l in range(tab.shape[0]):
        for c in range(tab.shape[1]):
            tab[l][c] = random.randint(0, 20)
    return tab


if __name__ == "__main__":
    '''
    test = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])
    tab_np = randomize_tab(4)
    print(tab_np)
    print("///////////////////////////")
    for line in range(tab_np.shape[0]):
        for col in range(tab_np.shape[1]):
            patch = tab_np[line:line + test.shape[0], col:col + test.shape[1]]
            print(patch)
            print(" / ")
    # print(tab_np)
    '''
    for i in range(0, 10, 2):
        print(i)
