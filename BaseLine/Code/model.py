# -*- coding: utf-8 -*-
"""
@Time    : 2019/10/22 13:39
@Author  : QuYue
@File    : model.py
@Software: PyCharm
Introduction:
"""
#%% Import Packages
import numpy as np

#%%
def add_one(unprocessed_X):
    return np.concatenate([unprocessed_X, np.ones([unprocessed_X.shape[0], 1])], axis=1)
#%%
class LR:
    # Logistic Regression
    def __init__(self, w):
        """
        init function
        :param w: float w
        """
        self.w = w # w


    def predict(self, X):
        """
        Predict
        :param X: np.array (n*m)  data n=data_number  m=feature_number
        :return:
               Y: np.array(n,)  predicted class 0 or 1  result of classification
               Y_Prob: np.array(n,)  predicted probability 0~1  probability of class 1
        """
        Y = np.zeros(len(X))
        Y_prob = np.zeros(len(X))
        for i, x in enumerate(X):
            y = 1/(1 + np.exp(-x.dot(self.w)))
            if y > 0.5:
                y_prob = 1
            else:
                y_prob = 0
            Y[i] = y
            Y[i] = y_prob
        return Y, Y_prob

    def loss(self, X, Y_real):
        """
        Loss
        :param X: np.array (n*m)  data n=data_number  m=feature_number
        :param Y_real: np.array(n,)  Real class 0 or 1  real label of data
        :return:
               L: float loss
        """
        L = 0
        for i in range(len(X)):
            x = X[i]
            y = Y_real[i]
            L += -y * x.dot(self.w) + np.log(1 + np.exp(x.dot(self.w)))
        return L

    def error(self, X, Y_real, Y=None):
        """
        error number
        :param X: np.array (n*m)  data n=data_number  m=feature_number
        :param Y_real: np.array(n,)  Real class 0 or 1  real label of data
        :param Y np.array(n,)(default=None)  predicted class 0 or 1  result of classification
        :return:
               w: float
               loss_list: list   The list of each epoch
        """
        if Y == None:
            Y, _ = self.predict(X)
        L = np.sum((Y - Y_real)**2)
        return L

    def fit(self, X, Y_real, C = 0.005, silent=False, mask=None):
        """
        Fit
        :param X: np.array (n*m)  data n=data_number  m=feature_number
        :param Y_real: np.array(n,)  Real class 0 or 1  real label of data
        :param C: float(default=0.005)  Gradient descent step length
        :param silent: bool(default=False) True: silent run  False: show each epoch's loss value.
        :param updateb: bool(default=True) True: update parameter b  False: not update
        :return:
               w: float w
               loss_list: list   The list of each epoch
        """
        if mask == None:
            mask = np.ones(X.shape[0])
        else:
            mask = np.array(mask)

        loss_list = []
        w_list = []
        for i in range(20):
            g_w = 0
            for j, x in enumerate(X):
                y = Y_real[j]
                p1 = np.exp(x.dot(self.w))/(1 + np.exp(x.dot(self.w)))
                g_w += -(x*(y - p1)).reshape(-1)

            self.w -= C * g_w * mask
            w_list.append(self.w.copy())
            loss_list.append(self.loss(X, Y_real))
            if silent == False:
                message = 'Epoch %s | Loss=%s | W=[' %(i, self.loss(X, Y_real))
                message += '%s,'*len(self.w) %tuple(self.w)
                message = message[:-1]
                message+= ']'
            print(message)
        return self.w, loss_list, w_list