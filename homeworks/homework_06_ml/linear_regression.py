#!/usr/bin/env python
# coding: utf-8
import numpy as np


class LinearRegression:
    def __init__(self, lambda_coef=1.0, regularization=None, alpha=0.5):
        """
        :param lambda_coef: constant coef for gradient descent step
        :param regularization: regularization type ("L1" or "L2") or None
        :param alpha: regularization coefficent
        """
        self.weights = np.array([])
        self.lambda_coef = lambda_coef
        self.regularization = regularization
        self.alpha = alpha
        self.loss = 0

    def fit(self, X_train, y_train):
        """
        Fit model using gradient descent method
        :param X_train: training data
        :param y_train: target values for training data
        :return: None
        """
        X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
        self.weights = np.ones((X_train.shape[1], 1))
        if self.regularization == 'L1':
            amend = self.alpha*np.sum(self.weights[1:, 0]**2)
            der_amend = 2*self.alpha*self.weights
        if self.regularization == 'L2':
            amend = self.alpha*np.sum(np.abs(self.weights[1:, 0]))
            der_amend = self.alpha*np.ones((self.weights.shape[0], 1))
        if self.regularization is None:
            amend = 0
            der_amend = np.zeros((self.weights.shape[0], 1))
        self.loss = np.sum((y_train - (X_train @ self.weights))**2)/y_train.shape[0] + amend
        iteration = 0
        while iteration <= 1000 and self.loss > 5:
            for i in np.arange(self.weights.shape[0]):
                self.weights[i, 0] -= (-2)*self.lambda_coef*np.sum((y_train - (X_train @ self.weights))*
                                                                   X_train[:, i].reshape(-1, 1) + der_amend[i, 0])
                self.loss = np.sum((y_train - (X_train @ self.weights)) ** 2) / y_train.shape[0] + amend
                iteration += 1

    def predict(self, X_test):
        """
        Predict using model.
        :param X_test: test data for predict in
        :return: y_test: predicted values
        """
        X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
        return X_test @ self.weights

    def get_weights(self):
        """
        Get weights from fitted linear model
        :return: weights array
        """
        return self.weights
