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
np.random.seed(10) # Set random seed

def target_fun(X):
    Y = np.zeros(len(X))
    w = np.array([5, -10])
    b = 50
    for i in range(len(X)):
        if np.sum(X[i] * w) + b > 0:
            Y[i] = 1
    return Y
w = np.array([5, -10])
b = 50
X1 = np.random.normal(loc=0, scale=3.0, size=[100,1]) # 数据集特征
X2 = np.zeros([len(X1), 1])
for i in range(len(X1)):
    X2[i] = X1[i] * (-w[0]/w[1]) + (-b/w[1])

X2 += np.random.normal(loc=0, scale=0.5, size=[100,1])
X = np.concatenate((X1,X2),axis=1)
Y = target_fun(X) # 数据集的标签

#%%
class LR:
    def __init__(self, w, b):
        self.w = w # w
        self.b = b # b

    def fun(self, X):
        # predict
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
        # loss
        L = 0
        for i in range(len(X)):
            x = X[i]
            y = Y_real[i]
            L += -y * (x.dot(self.w) + self.b) + np.log(1 + np.exp(x.dot(self.w) + self.b))
        return L

    def error_num(self, X, Y_real, Y = None):
        if Y == None:
            Y = self.fun(X)
        L = np.sum((Y - Y_real)**2)
        return L

    def fit(self, X, Y, C = 0.005):
        loss_list = []
        for i in range(500):
            g = 0
            for j, x in enumerate(X):
                y = Y[j]
                p1 = np.exp(x.dot(self.w)+self.b)/(1 + np.exp(x.dot(self.w)+self.b))
                g += -(x*(y - p1)).reshape(-1,1)

            self.w = self.w - C * g
            loss_list.append(self.loss(X, Y))
            print('第%s代 | Loss=%s | W=[%s, %s]' %(i,self.loss(X, Y),self.w[0], self.w[1]))
        return self.w, loss_list

#%%
w = np.random.normal(loc=0.0, scale=1.0, size=[2,1])
b = 50
lr = LR(w,b)
w,loss_list = lr.fit(X,Y)
print(w)
t = plt.plot(loss_list)
plt.show()

#%%
fig = plt.figure()
ax = Axes3D(fig)

W1 = np.arange(-10, 10, 0.5) + 5
W2 = np.arange(-10, 10, 0.5) - 10
L = np.zeros([len(W1),len(W2)])
for i, w1 in enumerate(W1):
    for j, w2 in enumerate(W2):
        w = np.array([w1, w2])
        lr = LR(w, 50)
        L[i,j] = lr.loss(X,Y)

W1, W2  = np.meshgrid(W1, W2)
ax.plot_surface(W1, W2, L, rstride=1, cstride=1, cmap='rainbow')

plt.show()

# #%%
# fig = plt.figure()
# W1 = np.arange(-1, 11,0.1)
# L = []
# for i in W1:
#     lr = LR(i, -11)
#     L.append(lr.loss(X, Y))
# L = np.array(L)
# plt.plot(W1,L)
# plt.show()
# print('Best w: %s' %W1[L.argmin()])











