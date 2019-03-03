#!/usr/bin/env python
# coding: utf-8


def calculate_determinant(list_of_lists):
    '''
    Метод, считающий детерминант входной матрицы,
    если это возможно, если невозможно, то возвращается
    None
    Гарантируется, что в матрице float
    :param list_of_lists: список списков - исходная матрица
    :return: значение определителя или None
    '''
    for i in range(len(list_of_lists)):
        if len(list_of_lists) != len(list_of_lists[i]):
            return None
    det = 0
    for l in range(len(list_of_lists)):
        if l % 2 == 0:
            sign = 1
        else:
            sign = -1
        m = []
        for i in range(1, len(list_of_lists)):
            m.append([])
            for j in range(0, len(list_of_lists[i])):
                if j == l:
                    continue
                else:
                    m[i - 1].append(list_of_lists[i][j])
        if len(m) == 1:
            det += sign * list_of_lists[0][l] * m[0][0]
        else:
            det += sign * list_of_lists[0][l] * calculate_determinant(m)
    return det
