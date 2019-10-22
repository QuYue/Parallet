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
import model

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
w = np.random.normal(loc=0.0, scale=1.0, size=[2])

b = 50
lr = model.LR(w,b)
w, b, loss_list = lr.fit(X,Y, silent=False, updateb=False)
print(w)
print(b)
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
        lr = model.LR(w, 50)
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











