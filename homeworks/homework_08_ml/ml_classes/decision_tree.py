#!/usr/bin/env python
# coding: utf-8

from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import log_loss


class DecisionTreeClassifier:
    '''
    Пишем свой велосипед - дерево для классификации
    '''

    def __init__(self, max_depth=None, min_leaf_size=0, max_leaf_number=None, min_inform_criter=None):
        '''
        Инициализируем наше дерево
        :param max_depth: один из возможных критерием останова - максимальная глубина дерева
        :param min_leaf_size: один из возможных критериев останова - число элементов в листе
        :param max_leaf_number: один из возможных критериев останова - число листов в дереве.
        Нужно подумать как нам отобрать "лучшие" листы
        :param min_inform_criter: один из критериев останова - процент прироста информации, который
        считаем незначительным
        '''
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.max_leaf_number = max_leaf_number
        self.min_inform_criter = min_inform_criter
        self.tree = []
        '''
            (feature, threshold, below_target, below_branch, above_target, above_branch, gain)
        '''
        self.leaves = 0

    @staticmethod
    def compute_split_information(x, y, threshold):
        '''
        Вспомогательный метод, позволяющий посчитать джини/энтропию для заданного разбиения
        :param X: Матрица (num_objects, 1) - срез по какой-то 1 фиче, по которой считаем разбиение
        :param y: Матрица (num_object, 1) - целевые переменные
        :param th: Порог, который проверяется
        :return: прирост информации
        '''
        more_count = 0
        less_count = 0
        for target in np.unique(y):
            less_count += (len(np.where(y[np.where(x <= threshold)] == target)[0])/\
                      y[np.where(x <= threshold)].shape[0])**2
            if y[np.where(x > threshold)].shape[0] == 0:
                more_count = 1
            else:
                more_count += (len(np.where(y[np.where(x > threshold)] == target)[0])/\
                          y[np.where(x > threshold)].shape[0])**2
        return (1 - more_count)*len(np.where(x > threshold)[0])/x.shape[0],\
               (1 - less_count)*len(np.where(x <= threshold)[0])/x.shape[0]

    def fit(self, x, y, depth=1, prev_gain=0, prev_branch=None, prev_leaf='both', offset=None):
        '''
        Стандартный метод обучения
        :param X: матрица объекто-признаков (num_objects, num_features)
        :param y: матрица целевой переменной (num_objects, 1)
        :return: None
        '''
        if offset is None:
            offset = list(range(x.shape[1]))
        max_gain = 1
        best_feature = 0
        best_threshold = 0
        best_above = 0
        best_below = 0
        for j in range(x.shape[1]):
            for threshold in np.unique(x[:, j]):
                above, below = self.compute_split_information(x[:, j], y, threshold)
                if above + below < max_gain:
                    max_gain = above + below
                    best_above, best_below = above, below
                    best_feature = j
                    best_threshold = threshold
        if prev_branch is not None and self.tree[prev_branch][2] is None and prev_leaf == 'below':
            self.tree[prev_branch][3] = len(self.tree)
        if prev_branch is not None and self.tree[prev_branch][4] is None and prev_leaf == 'above':
            self.tree[prev_branch][5] = len(self.tree)
        prev_branch = len(self.tree)
        if x.shape[1] != 1:
            if best_below == 0 and best_above != 0:
                self.leaves += 1
                self.tree.append([offset[best_feature], best_threshold,
                                  y[np.where(x[:, best_feature] <= best_threshold)][0],
                                  None, None, None, best_above + best_below])
                offset.pop(best_feature)
                if self.max_depth is None or depth < self.max_depth:
                    if len(np.where(x[:, best_feature] >= best_threshold)[0]) >= self.min_leaf_size:
                        x_new = self.slice(x, best_feature, best_threshold, 'above')
                        y_new = y[np.where(x[:, best_feature] > best_threshold)]
                        self.fit(x_new, y_new, depth=depth + 1, prev_gain=prev_gain - max_gain,
                                 prev_branch=prev_branch, prev_leaf='above', offset=offset.copy())
                    else:
                        self.tree[-1][4] = int(np.mean(y[np.where(x[:, best_feature] >= best_threshold)]))
                else:
                    self.tree[-1][4] = int(np.mean(y[np.where(x[:, best_feature] >= best_threshold)]))
            elif best_below != 0 and best_above == 0:
                self.leaves += 1
                self.tree.append([offset[best_feature], best_threshold, None, None,
                                  y[np.where(x[:, best_feature] > best_threshold)][0], None, best_above + best_below])
                offset.pop(best_feature)
                if self.max_depth is None or depth < self.max_depth:
                    if len(np.where(x[:, best_feature] <= best_threshold)[0]) >= self.min_leaf_size:
                        x_new = self.slice(x, best_feature, best_threshold, 'below')
                        y_new = y[np.where(x[:, best_feature] <= best_threshold)]
                        self.fit(x_new, y_new, depth=depth + 1, prev_gain=prev_gain - max_gain,
                                 prev_branch=prev_branch, prev_leaf='below', offset=offset.copy())
                    else:
                        self.tree[-1][2] = int(np.mean(y[np.where(x[:, best_feature] <= best_threshold)]))
                else:
                    self.tree[-1][2] = int(np.mean(y[np.where(x[:, best_feature] <= best_threshold)]))
            elif best_below != 0 and best_above != 0:
                self.tree.append([offset[best_feature], best_threshold, None, None, None, None,
                                  best_above + best_below])
                offset.pop(best_feature)
                if self.max_depth is None or depth < self.max_depth:
                    if len(np.where(x[:, best_feature] <= best_threshold)[0]) >= self.min_leaf_size:
                        x_new = self.slice(x, best_feature, best_threshold, 'below')
                        y_new = y[np.where(x[:, best_feature] <= best_threshold)]
                        self.fit(x_new, y_new, depth=depth + 1, prev_gain=prev_gain - max_gain,
                                 prev_branch=prev_branch, prev_leaf='below', offset=offset.copy())
                    else:
                        self.tree[-1][2] = int(np.mean(y[np.where(x <= best_threshold)]))
                    if len(np.where(x[:, best_feature] > best_threshold)[0]) >= self.min_leaf_size:
                        x_new = self.slice(x, best_feature, best_threshold, 'above')
                        y_new = y[np.where(x[:, best_feature] > best_threshold)]
                        self.fit(x_new, y_new, depth=depth + 1, prev_gain=prev_gain - max_gain,
                                 prev_branch=prev_branch, prev_leaf='above', offset=offset.copy())
                    else:
                        self.tree[-1][4] = int(np.mean(y[np.where(x > best_threshold)]))
                else:
                    self.tree[-1][2] = int(np.mean(y[np.where(x[:, best_feature] <= best_threshold)]))
                    self.tree[-1][4] = int(np.mean(y[np.where(x[:, best_feature] > best_threshold)]))
            elif best_below == 0 and best_above == 0:
                self.leaves += 2
                self.tree.append([offset[best_feature], best_threshold,
                                  y[np.where(x[:, best_feature] <= best_threshold)][0],
                                  None, y[np.where(x[:, best_feature] > best_threshold)][0], None,
                                  best_above + best_below])
                offset.pop(best_feature)
        else:
            self.tree.append([offset[best_feature], best_threshold,
                              int(np.mean(y[np.where(x[:, best_feature] <= best_threshold)])),
                              None, int(np.mean(y[np.where(x[:, best_feature] > best_threshold)])), None,
                              best_above + best_below])

    def predict(self, x):
        '''
        Метод для предсказания меток на объектах X
        :param X: матрица объектов-признаков (num_objects, num_features)
        :return: вектор предсказаний (num_objects, 1)
        '''
        result = np.array([], dtype=int)
        for i in range(x.shape[0]):
            result = np.append(result, self.make_pred(x[i, :]))
        return result

    def predict_proba(self, x):
        '''
        метод, возвращающий предсказания принадлежности к классу
        :param X: матрица объектов-признаков (num_objects, num_features)
        :return: вектор предсказанных вероятностей (num_objects, 1)
        '''
        raise NotImplementedError

    @staticmethod
    def slice(x, col, threshold, condition):
        result = np.ones((1, x.shape[1]))
        if condition == 'above':
            for row in range(x.shape[0]):
                if x[row, col] > threshold:
                    result = np.concatenate((result, x[row, :].reshape(1, -1)), axis=0)
        if condition == 'below':
            for row in range(x.shape[0]):
                if x[row, col] <= threshold:
                    result = np.concatenate((result, x[row, :].reshape(1, -1)), axis=0)
        return np.delete(np.delete(result, 0, axis=0), col, axis=1)

    def make_pred(self, x, branch=0):
        if x[self.tree[branch][0]] <= self.tree[branch][1]:
            if self.tree[branch][2] is None:
                result = self.make_pred(x, self.tree[branch][3])
            else:
                result = self.tree[branch][2]
        else:
            if self.tree[branch][4] is None:
                result = self.make_pred(x, self.tree[branch][5])
            else:
                result = self.tree[branch][4]
        return result
