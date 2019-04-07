#!/usr/bin/env python
# coding: utf-8


import numpy as np


class CSRMatrix:
    """
    CSR (2D) matrix.
    Here you can read how CSR sparse matrix works: https://en.wikipedia.org/wiki/Sparse_matrix
    """
    def __init__(self, init):
        """
        :param init: can be usual dense matrix
        or
        (row_ind, col, data) tuple with np.arrays,
            where data, row_ind and col_ind satisfy the relationship:
            a[row_ind[k], col_ind[k]] = data[k]
        """
        self.A = np.array([])
        self.IA = np.array([0])
        self.JA = np.array([])
        if isinstance(init, tuple) and len(init) == 3:
            matrix = []
            for i in np.lexsort((init[2], init[1], init[0])):
                matrix.append([init[2][i], init[1][i], init[0][i]])
            matrix = np.array(matrix)
            self.A = np.append(self.A, matrix[:, 0])
            self.JA = np.append(self.JA, matrix[:, 1])
            sum = 0
            for i in np.arange(matrix.shape[0]):
                for j in np.arange(matrix.shape[0]):
                    if matrix[j, 2] == i:
                       sum += 1
                self.IA = np.append(self.IA, sum)
        elif isinstance(init, np.ndarray):
            self.A = np.append(self.A, init[np.nonzero(init)])
            for i in np.arange(init.shape[0]):
                self.JA = np.append(self.JA, np.nonzero(init[i, :]))
            sum = 0
            for i in np.arange(init.shape[0]):
                sum += init[i, :][np.nonzero(init[i, :])].shape[0]
                self.IA = np.append(self.IA, sum)
        else:
            raise ValueError

    def get_item(self, i, j):
        """
        Return value in i-th row and j-th column.
        Be careful, i and j may have invalid values (-1 / bigger that matrix size / etc.).
        """
        for k in np.arange(self.A.shape[0]):
            if (self.JA[k] == j) & (k in np.arange(self.IA[i], self.IA[i + 1])):
                return self.A[k]
        return 0

    def set_item(self, i, j, value):
        """
        Set the value to i-th row and j-th column.
        Be careful, i and j may have invalid values (-1 / bigger that matrix size / etc.).
        """
        found = False
        for k in np.arange(self.A.shape[0]):
            if (self.JA[k] == j) & (k in np.arange(self.IA[i], self.IA[i + 1])):
                self.A[k] = value
                found = True
                break
        if not found:
            if self.IA[i + 1] - self.IA[i] == 0:
                self.A = np.insert(self.A, self.IA[i + 1], value)
                self.JA = np.insert(self.JA, self.IA[i + 1], j)
                self.IA[i + 1:] += 1
            for k in np.arange(self.IA[i], self.IA[i + 1]):
                if (j < self.JA[k]) | (k == np.arange(self.IA[i], self.IA[i + 1])[-1]):
                    self.A = np.insert(self.A, k + 1, value)
                    self.JA = np.insert(self.JA, k + 1, j)
                    self.IA[i + 1:] += 1
                    break

    def to_dense(self):
        """
        Return dense representation of matrix (2D np.array).
        """
        max_row_len = int(self.IA.shape[0] - 1)
        max_col_len = int(np.amax(self.JA) + 1)
        result = np.empty([max_row_len, max_col_len])
        for i in np.arange(max_row_len):
            for j in np.arange(max_col_len):
                result[i, j] = self.get_item(i, j)
        return result


# zero_matrix = np.zeros((5, 5))
# matrix = np.random.randint(0, 2, (5, 5))
# print(matrix)
# csr_matrix = CSRMatrix(matrix)
# print(csr_matrix.to_dense())
# for i, j in zip(range(matrix.shape[0]), range(matrix.shape[1])):
#     csr_matrix.set_item(i, j, matrix[i, j])
# print(csr_matrix.A)
# for i in range(5):
#     for j in range(5):
#         print(csr_matrix.get_item(i, j), end=' ')
#     print('')
# a = np.array([[0, 0, 0, 0],
#               [5, 8, 0, 0],
#               [0, 0, 3, 0],
#               [0, 6, 0, 0]])
# test_1 = CSRMatrix(a)
# print(test_1.A)
# print(test_1.JA)
# print(test_1.IA)
#
# test_2 = CSRMatrix(([1, 1, 2, 3],
#                  [0, 1, 2, 1],
#                  [5, 8, 3, 6]))
# print(test_2.A)
# print(test_2.JA)
# print(test_2.IA)
#
# for i in range(4):
#     for j in range(4):
#         print(test_2.get_item(i, j), end=' ')
#     print('')
#
# test_1.set_item(2, 3, 10)
# print(test_1.A)
# print(test_1.JA)
# print(test_1.IA)
#
# for i in range(4):
#     for j in range(4):
#         print(test_1.get_item(i, j), end=' ')
#     print('')
# print(test_1.to_dense())


# matrix = np.random.randint(0, 2, (5, 5))
# print(matrix)
# csr_matrix = CSRMatrix(matrix)
# print(csr_matrix.to_dense())
# csr_matrix.set_item(1, 1, 20)
# for i in range(5):
#     for j in range(5):
#         print(csr_matrix.get_item(i, j), end=' ')
#     print('')


