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
        self.A = np.array([], dtype=int)
        self.IA = np.array([0], dtype=int)
        self.JA = np.array([], dtype=int)
        if isinstance(init, tuple) and len(init) == 3:
            self.A = np.append(self.A, init[2])
            self.JA = np.append(self.JA, init[1])
            self.IA = np.zeros(init[0][-1] + 2, dtype=int)
            for i in init[0]:
                self.IA[i + 1:] += 1
        elif isinstance(init, np.ndarray):
            self.A = np.append(self.A, init[np.nonzero(init)])
            self.JA = np.append(self.JA, np.nonzero(init)[1])
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
        for k in np.arange(self.IA[i], self.IA[i + 1]):
            if self.JA[k] == j:
                return self.A[k]
        return 0

    def set_item(self, i, j, value):
        """
        Set the value to i-th row and j-th column.
        Be careful, i and j may have invalid values (-1 / bigger that matrix size / etc.).
        """
        found = False
        for k in np.arange(self.IA[i], self.IA[i + 1]):
            if self.JA[k] == j:
                self.A[k] = value
                found = True
                break
        if not found:
            if self.IA[i + 1] - self.IA[i] == 0:
                self.A = np.insert(self.A, self.IA[i + 1], value)
                self.JA = np.insert(self.JA, self.IA[i + 1], j)
                self.IA[i + 1:] += 1
            for k in np.arange(self.IA[i], self.IA[i + 1]):
                if j < self.JA[k] or k == np.arange(self.IA[i], self.IA[i + 1])[-1]:
                    self.A = np.insert(self.A, k + 1, value)
                    self.JA = np.insert(self.JA, k + 1, j)
                    self.IA[i + 1:] += 1
                    break

    def to_dense(self):
        """
        Return dense representation of matrix (2D np.array).
        """
        max_row_len = self.IA.shape[0] - 1
        max_col_len = np.amax(self.JA) + 1
        result = np.zeros([max_row_len, max_col_len])
        for i in np.arange(max_row_len):
            for j in np.arange(self.IA[i], self.IA[i + 1]):
                result[i, self.JA[j]] = self.A[j]
        return result
