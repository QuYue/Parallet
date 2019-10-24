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
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
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
X0 = np.concatenate((X1,X2),axis=1)
Y = target_fun(X0) # 数据集的标签

#%%
X = model.add_one(X0)
w = np.random.normal(loc=0.0, scale=1.0, size=[3])
w[2] = 50
lr = model.LR(w)
w, loss_list, w_list = lr.fit(X,Y, silent=False, mask=[1,1,0])
print(w)
t = plt.plot(loss_list)
plt.show()


#%%
edge = []
x = []
y = []
z = []
l = len(w_list)
for i in range(0,l-1):
    edge.append([[w_list[i][0], w_list[i][1], loss_list[i]+100], [w_list[i+1][0], w_list[i+1][1], loss_list[i+1]+100]])
    x.append(w_list[i][0])
    y.append(w_list[i][1])
    z.append(loss_list[i]+100)

#%%
fig = plt.figure()
ax = Axes3D(fig)

W1 = np.arange(-10, 10, 1) + 5
W2 = np.arange(-10, 10, 1) - 10
L = np.zeros([len(W1),len(W2)])
for i, w1 in enumerate(W1):
    for j, w2 in enumerate(W2):
        w = np.array([w1, w2, 50])
        lr = model.LR(w)
        L[i,j] = lr.loss(X,Y)

W2, W1 = np.meshgrid(W2, W1)
ax.scatter(x,y,z,alpha=1)
ax.plot(x,y,z)
ax.add_collection3d(Line3DCollection(edge, colors='k', linewidths=2, linestyles='-'))
ax.plot_surface(W1, W2, L, rstride=1, cstride=1, cmap='rainbow')
# ax.plot_wireframe(W1, W2, L, rstride=1, cstride=1, alpha=0.1)
# ax.contour(W1, W2, L, cmap=cm.Accent, linewidths=2)

ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('loss')
# cset = ax.contour(W1, W2, L, zdir='z', offset=L.min()-1, cmap=cm.coolwarm)
# cset = ax.contour(W1, W2, L, zdir='x', offset=W1.min()-1,cmap=cm.coolwarm)
# cset = ax.contour(W1, W2, L, zdir='y', offset=W2.max()+1,cmap=cm.coolwarm)




plt.show()

#











