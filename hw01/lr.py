#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    @desc 手写的基于梯度下降的线性回归
    @author liuxiji
    @time 2020/6/3 3:57 下午
"""

import numpy as np


class LinearRegression(object):
    """ 线性回归 """

    def __init__(self):
        self.name = "LR"
        self.__theta = None

    def h(self, theta, X):
        """
        h = X * theta

        :param theta: 参数是个N维向量
        :param X: 训练集矩阵
        :return: y, len(y)维的向量, 预测结果的N维向量
        """
        return np.dot(X, theta)

    def loss(self, theta, X, y):
        """
        损失函数, 均方误差 l = 1/2m * sum( (h(x) - y) ^ 2 ); m = len(y), 样本的数量

        :param theta: 参数向量
        :param X: 训练集矩阵
        :param y: 训练集label向量
        :return: loss, float, 当前theta下的误差
        """
        return np.sum((self.h(theta, X) - y) ** 2) / (2 * len(y))

    def dl(self, theta, X, y):
        """
        loss的偏导的矩阵表示算法: dl = Xt * (X * theta - y) / len(y)

        :param theta: 参数向量
        :param X: 训练集矩阵
        :param y: 训练集label向量
        :return: dl, len(y)维的向量
        """
        vdl = X.T.dot(X.dot(theta) - y) / len(y)
        return vdl

    def gradient_decent(self, X, y, initial_theta, eta=0.01, max_iters=10000, episode=1e-8):
        """
        梯度下降函数

        :param X: 训练集矩阵
        :param y: 训练集label向量
        :param initial_theta: 初始化的theta
        :param eta: learning rate
        :param max_iters: 最大迭代次数
        :param episode: 容忍度
        :return: 最终的theta向量
        """

        theta = initial_theta
        for i in range(int(max_iters)):
            last_theta = theta

            gradient = self.dl(theta, X, y)
            theta = last_theta - eta * gradient

            # 更新theta以后，loss没有什么变化
            if abs(self.loss(theta, X, y) - self.loss(last_theta, X, y)) <= episode:
                break

        return theta

    def fit(self, X, y, eta=0.01, n_iters=100000):
        """
        利用gradient decent来求theta

        :param X: 训练集矩阵
        :param y: 训练集label向量
        :param eta: learning rate
        :param n_iters: 最大迭代次数
        :return: self
        """

        # 为X增加一列X0，并置为1（方便匹配w0） X: m * n 矩阵 (m行数据，每行数据n维)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # 初始化theta为全0 （也可以random取值) theta: n * 1 向量
        initial_theta = np.zeros((X.shape[1], 1))

        self.__theta = self.gradient_decent(X, y, initial_theta, eta, n_iters)

        return self

    def predict(self, Xn):
        """
        predict y by Xn

        :param Xn:
        :return:
        """
        X = np.hstack([np.ones((Xn.shape[0], 1)), Xn])
        return self.h(self.__theta, X)

    def fit_normal(self, X_train, y_train):
        # 根据训练数据集X_train, y_train训练Linear Regression模型
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 对训练数据集添加 bias
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self.__theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        # self.intercept_ = self._theta[0]
        # self.coef_ = self._theta[1:]

        return self
