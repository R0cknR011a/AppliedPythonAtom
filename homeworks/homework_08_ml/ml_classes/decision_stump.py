#!/usr/bin/env python
# coding: utf-8
import numpy as np


class DecisionStumpRegressor:
    '''
    Класс, реализующий решающий пень (дерево глубиной 1)
    для регрессии. Ошибку считаем в смысле MSE
    '''

    def __init__(self):
        '''
        Мы должны создать поля, чтобы сохранять наш порог th и ответы для
        x <= th и x > th
        '''
        self.threshold = 0
        self.above = 0
        self.below = 0

    def fit(self, x, y):
        '''
        метод, на котором мы должны подбирать коэффициенты th, y1, y2
        :param X: массив размера (1, num_objects)
        :param y: целевая переменная (1, num_objects)
        :return: None
        '''
        result = np.array([])
        sort = x.argsort()
        x = x[0, sort[0]]
        y = y[0, sort[0]]
        for i in range(1, x.shape[0]):
            index = (np.var(y[:i])*i + (y.shape[0] - i)*np.var(y[i:]))/y.shape[0]
            result = np.append(result, index)
        min_index = np.argmin(result)
        self.threshold = x[min_index]
        self.above = np.mean(y[:min_index])
        self.below = np.mean(y[min_index:])

    def predict(self, X):
        '''
        метод, который позволяет делать предсказания для новых объектов
        :param X: массив размера (1, num_objects)
        :return: массив, размера (1, num_objects)
        '''
        result = np.ones((X.shape[0], X.shape[1]))
        for i in X[0, :]:
            if i <= self.threshold:
                result[0, i] = self.below
            else:
                result[0, i] = self.above
        return result
