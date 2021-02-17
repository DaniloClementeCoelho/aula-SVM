import math
import numpy as np
import matplotlib.pyplot as plt

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

meu_x1 = np.copy(Y)
meu_x2 = np.zeros(len(Y))
target = np.zeros(len(Y))
f_de_meux1 = np.zeros(3)
i = 0

for k in Y:
    f_de_meux1 = X[Y == k]
    meu_x2[i] = np.random.uniform(-3, 3)
    if len(f_de_meux1) == 1:
        if np.abs(meu_x2[i]) < np.abs(f_de_meux1[0]):
            target[i] = 1
        else:
            target[i] = 0
    elif len(f_de_meux1) == 2:
         if meu_x2[i] < max(f_de_meux1[0], f_de_meux1[1]) and meu_x2[i] > min(f_de_meux1[0], f_de_meux1[1]):
            target[i] = 1
         else:
            target[i] = 0
    elif len(f_de_meux1) == 3:
         if meu_x2[i] < max(f_de_meux1[0], f_de_meux1[1], f_de_meux1[2]) and meu_x2[i] > min(f_de_meux1[0], f_de_meux1[1], f_de_meux1[2]):
            target[i] = 1
         else:
            target[i] = 0
    else:
        target[i] = 99
    i=i+1



fig = plt.figure()
ax1  = fig.add_subplot(211)
ax1.scatter(Y, X, c='black', s=2)
ax2  = fig.add_subplot(212)
ax2.scatter(meu_x1, meu_x2, c=target)

plt.grid()
plt.show()

'''
db = pd.DataFrame({'x1': meu_x1, 'x2': meu_x2, 'target': target})
db.to_excel("./Bases/BATMAN.xlsx")
'''