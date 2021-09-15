import numpy as np


def createMatrix(row, column):
    # return np.random.randint(0, 2, (row, column))
    return np.matrix([[1, 1], [1, 2]])


def swapRows(matrix, row1, row2):
    matrix[[row1, row2]] = matrix[[row2, row1]]


def SumRows(matrix, row, rowsum):
    matrix[row] = np.array([((a + k) % 2) for a, k in zip(matrix[row], matrix[rowsum])])


if __name__ == '__main__':
    v = createMatrix(2, 2)
    print(v)
    swapRows(v, 0, 1)
    print(v)
    SumRows(v, 0, 1)
    # print([(a+k) for a, k in zip(v[1], v[0])])
    print(v)
