import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import math

# CALCULA x E y DO GABARITO DO GRÁFICO PARA PLOTAR N O FINAL
Y = np.arange(-4, 4, .01)
X = np.zeros((0))
for y in Y:
    X = np.append(X, abs(y / 2) - 0.09137 * y ** 2 + math.sqrt(1 - (abs(abs(y) - 2) - 1) ** 2) - 3)

Y1 = np.append(np.arange(-7, -3.01, .01), np.arange(3, 7.01, .01))
X1 = np.zeros((0))
for y in Y1:
    X1 = np.append(X1, 3 * math.sqrt(-(y / 7) ** 2 + 1))
X = np.append(X, X1)
Y = np.append(Y, Y1)
Y1 = np.append(np.arange(-7., -4.01, .01), np.arange(4, 7.01, .01))
X1 = np.zeros((0))
for y in Y1:
    X1 = np.append(X1, -3 * math.sqrt(-(y / 7) ** 2 + 1))
X = np.append(X, X1)
Y = np.append(Y, Y1)
Y1 = np.append(np.arange(-1, -.81, .01), np.arange(.8, 1.01, .01))
X1 = np.zeros((0))
for y in Y1:
    X1 = np.append(X1, 9 - 8 * abs(y))
X = np.append(X, X1)
Y = np.append(Y, Y1)
Y1 = np.arange(-.5, .51, .01)
X1 = np.zeros((0))
for y in Y1:
    X1 = np.append(X1, 2)
X = np.append(X, X1)
Y = np.append(Y, Y1)
Y1 = np.append(np.arange(-2.9, -1.01, .01), np.arange(1, 2.91, .01))
X1 = np.zeros((0))
for y in Y1:
    X1 = np.append(X1, 1.5 - .5 * abs(y) - 1.89736 * (math.sqrt(3 - y ** 2 + 2 * abs(y)) - 2))
X = np.append(X, X1)
Y = np.append(Y, Y1)
Y1 = np.append(np.arange(-.7, -.46, .01), np.arange(.45, .71, .01))
X1 = np.zeros((0))
for y in Y1:
    X1 = np.append(X1, 3 * abs(y) + .75)
X = np.append(X, X1)
Y = np.append(Y, Y1)

for j in range(len(Y)):
    Y[j] = round(Y[j], 3)



# ###############     INICIO DO PROGRAMA DE AJUSTE DO MODELO

base = pd.read_excel("../Bases/BATMAN.xlsx")
# base = pd.read_excel("./Bases/BATMAN.xlsx") # para rodar no console python
explicativas = base.iloc[:, 1:3]

SVM = svm.SVC(kernel='rbf')
# SVM = svm.SVC(kernel='linear')
parametros = {'C': [1, 10, 100, 1000], 'gamma': [100, 10, 1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(estimator=SVM, param_grid=parametros)
modelo = grid.fit(explicativas, base.target)
# modelo.best_params_


x1_surf, x2_surf = np.meshgrid(np.arange(-10, 10, 1),
                               np.arange(-10, 10, 1))
x1 = x1_surf.ravel()
x2 = x2_surf.ravel()
X12 = np.concatenate([x1.reshape(x1.size, 1), x2.reshape(x1.size, 1)], axis=1)
prev_svm = modelo.predict(X12).reshape(x1_surf.shape)

figura = plt.figure()
ax = figura.add_subplot(111)
ax.set_xlim(-7, 7)
ax.set_ylim(-3, 3)
# ax.scatter(SVM.support_vectors_[:,0], SVM.support_vectors_[:,1], facecolors='none', edgecolors='black', s=100)
ax.scatter(base['x1'], base['x2'], c=base['target'], alpha=0.1)
ax.scatter(Y, X, c='black')
ax.contourf(x1_surf, x2_surf, prev_svm, cmap=plt.cm.coolwarm, alpha=0.5)
plt.show()