# -*- coding: utf-8 -*-
"""
@Time    : 2019/10/19 18:06
@Author  : QuYue
@File    : main.py
@Software: PyCharm
Introduction:
"""


#%% Import Package
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%
np.random.seed(10)
def target_fun(X):
    Y = np.zeros(len(X))
    w = np.array([0.2])
    b = 1
    for i in range(len(X)):
        if np.sum(X[i] * w + b)> 0:
            Y[i] = 1
    return Y


X = np.random.normal(loc=0.0, scale=2.0, size=[100,1])
Y = target_fun(X)
#%%
w = np.random.normal(loc=0.0, scale=1.0, size=[1,1])
b = 1
alpha=20
#%%
class LR:
    def __init__(self, w, b):
        self.w = w
        self.b = b
    def fun(self, X):
        Y = np.zeros(len(X))
        for i, x in enumerate(X):
            y = 1/(1 + np.exp(-(x.dot(self.w) + self.b)))
            if y > 0.5:
                y = 1
            else:
                y = 0
            Y[i] = y

        return Y

    def loss(self, X, Y_real):
        L = 0
        for i in range(len(X)):
            x = X[i]
            y = Y_real[i]
            L += -y*(x.dot(self.w)+self.b) + np.log10(1 + np.exp(x.dot(self.w)+self.b))
        return L

    def loss_norm(self, X, Y_real):
        L = 0
        for i in range(len(X)):
            x = X[i]
            y = Y_real[i]
            L += -y*(x.dot(self.w)+self.b) + np.log10(1 + np.exp(x.dot(self.w)+self.b))
        L +=  alpha * np.linalg.norm(self.w,2)
        return L

    def loss2(self, X, Y_real, Y = None):
        if Y == None:
            Y = self.fun(X)
        L = np.sum((Y - Y_real)**2)
        return L

    def get_value(self, X, Y, C = 0.004):
        loss_list = []
        for i in range(100):
            g = 0
            for j, x in enumerate(X):
                y = Y[j]
                p1 = np.exp(x.dot(self.w))/(1 + np.exp(x.dot(self.w)))
                g += -(x*(y-p1)).reshape(-1,1)
            g += 2 * alpha * self.w
                # g[2] = 0
            self.w = self.w - C * g
            loss_list.append(self.loss(X, Y))
            print('第%s代 | Loss=%s | W=[%s,%s,%s]' %(i,self.loss(X, Y),self.w[0], self.w[1], self.w[2]))
        return self.w, loss_list



#%%
# w = np.random.normal(loc=0.0, scale=1.0, size=[3,1])
# # w[0]=0.1
# # w[1]=0.2
# # w[2] = 0.3
# b = 0
# lr = LR(w,b)
# w,loss_list = lr.get_value(X,Y)
# w3 = w[2]
# print(w)
# t = plt.plot(loss_list)
# plt.show()

#%%

# fig = plt.figure()
# ax = Axes3D(fig)
#
# W1 = np.arange(-1, 5, 0.05)
# W2 = np.arange(-1, 5, 0.05)
# L = np.zeros([len(W1),len(W2)])
# for i, w1 in enumerate(W1):
#     for j, w2 in enumerate(W2):
#         w = np.array([w1, w2, w3])
#         lr = LR(w,0)
#         L[i,j] = lr.loss(X,Y)
#
# W1, W2  = np.meshgrid(W1, W2)
# ax.plot_surface(W1, W2, L, rstride=1, cstride=1, cmap='rainbow')
#
# plt.show()
#%%
fig = plt.figure()
W1 = np.arange(-1, 3, 0.01)
L = []
for i in W1:
    lr = LR(i, 1)
    L.append(lr.loss2(X, Y))
plt.plot(L)
plt.show()











