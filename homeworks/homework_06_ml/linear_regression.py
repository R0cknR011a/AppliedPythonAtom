#!/usr/bin/env python
# coding: utf-8
import numpy as np


class LinearRegression:
    def __init__(self, lambda_coef=1.0, regularization=None, alpha=0.5, n_iter=1000, eps=5):
        """
        :param lambda_coef: constant coef for gradient descent step
        :param regularization: regularization type ("L1" or "L2") or None
        :param alpha: regularization coefficent
        """
        self.weights = np.array([])
        self.lambda_coef = lambda_coef
        self.regularization = regularization
        self.alpha = alpha
        self.n_iter = n_iter
        self.eps = eps
        self.learned = False

    def fit(self, x_train, y_train):
        """
        Fit model using gradient descent method
        :param x_train: training data
        :param y_train: target values for training data
        :return: None
        """
        amend = 0
        der_amend = 0
        x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
        self.weights = np.ones((x_train.shape[1], 1))
        if self.regularization == 'L1':
            amend = self.alpha*np.sum(self.weights[1:, 0]**2)
            der_amend = 2*self.alpha*self.weights
        elif self.regularization == 'L2':
            amend = self.alpha*np.sum(np.abs(self.weights[1:, 0]))
            der_amend = self.alpha*np.ones((self.weights.shape[0], 1))
        elif self.regularization is None:
            amend = 0
            der_amend = np.zeros((self.weights.shape[0], 1))
        loss = np.sum((y_train - (x_train @ self.weights)) ** 2) / y_train.shape[0] + amend
        iteration = 0
        while iteration <= self.n_iter and loss > self.eps:
            for i in np.arange(self.weights.shape[0]):
                self.weights[i, 0] -= (-2)*self.lambda_coef*np.sum((y_train - (x_train @ self.weights)) *
                                                                   x_train[:, i].reshape(-1, 1) + der_amend[i, 0])
                loss = np.sum((y_train - (x_train @ self.weights)) ** 2) / y_train.shape[0] + amend
                iteration += 1
        self.learned = True

    def predict(self, x_test):
        """
        Predict using model.
        :param x_test: test data for predict in
        :return: y_test: predicted values
        """
        if self.learned:
            x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
            return x_test @ self.weights

    def get_weights(self):
        """
        Get weights from fitted linear model
        :return: weights array
        """
        if self.learned:
            return self.weights
