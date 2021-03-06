import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def plota_corte(w, i):
    beta = np.array([-w[0]/w[2], -w[1]/w[2]])
    x_reta = np.array([(1, 1), (1, 7)])
    ax.plot(x_reta[:, 1], np.dot(x_reta, beta), label=i)

# ########################### DADOS #########################
raw_data = [(
1.0000,2.6487,4.5192,
1.0000,1.5438,2.4443,
1.0000,1.8990,4.2409,
1.0000,2.4711,5.8097,
1.0000,3.3590,6.4423,
1.0000,3.2406,5.8097,
1.0000,3.8128,6.3917,
1.0000,4.4441,6.8725,
1.0000,3.6747,6.7966,
1.0000,4.7401,8.1630,
1.0000,3.8917,7.4038,
1.0000,4.6020,7.6316,
1.0000,5.7265,7.7581,
1.0000,4.9571,6.5688,
1.0000,3.9903,5.3543,
1.0000,3.0236,4.4686,
1.0000,2.0568,2.9757,
1.0000,1.2676,2.4443,
1.0000,1.1690,0.9008,
1.0000,1.7411,2.1154,
1.0000,1.3860,3.2794,
1.0000,1.5636,4.1650,
1.0000,1.8793,4.8482,
1.0000,2.7868,3.3300,
1.0000,3.5563,5.1518,
1.0000,4.0693,6.2652,
1.0000,4.3849,6.2652,
1.0000,1.5438,7.2014,
1.0000,2.4120,7.6569,
1.0000,1.7806,6.1387,
1.0000,1.4057,4.4939,
1.0000,2.6093,4.8735,
1.0000,3.0828,5.5314,
1.0000,3.9311,6.0121,
1.0000,4.7598,7.1508,
1.0000,5.3122,7.7075,
1.0000,5.7068,8.3148,
1.0000,5.1149,8.5172,
1.0000,5.4109,8.7449,
1.0000,3.8128,7.8593,
1.0000,3.2406,6.9990,
1.0000,2.9052,5.5061,
1.0000,2.6882,4.9241,
1.0000,3.8325,6.6447,
1.0000,4.5428,7.6822,
1.0000,5.7857,8.0364,
1.0000,6.5552,8.9221,
1.0000,5.2530,7.8593,
1.0000,5.2333,6.5941,
1.0000,4.7598,6.0374,
-1.0000,4.5822,2.7227,
-1.0000,3.6549,1.9383,
-1.0000,2.9841,1.6852,
-1.0000,4.4244,4.3168,
-1.0000,3.7536,3.4312,
-1.0000,5.2728,5.4808,
-1.0000,4.8387,4.1144,
-1.0000,4.4244,3.2034,
-1.0000,5.3911,4.1144,
-1.0000,6.0817,5.1012,
-1.0000,5.5687,4.8988,
-1.0000,6.4565,5.9615,
-1.0000,6.0028,5.7591,
-1.0000,6.7722,6.6953,
-1.0000,6.6538,5.7338,
-1.0000,7.1471,6.6194,
-1.0000,7.5219,7.2014,
-1.0000,6.8314,7.2014,
-1.0000,7.6206,8.5931,
-1.0000,7.1865,7.7581,
-1.0000,7.7784,7.7581,
-1.0000,7.6009,5.1012,
-1.0000,6.4960,4.2156,
-1.0000,5.8055,3.4818,
-1.0000,5.0163,2.3684,
-1.0000,4.1876,1.7864,
-1.0000,3.4379,0.9008,
-1.0000,5.7857,0.9008,
-1.0000,6.3382,1.9636,
-1.0000,4.9571,1.4069,
-1.0000,6.8511,2.4190,
-1.0000,6.0817,2.8745,
-1.0000,7.1668,4.0132,
-1.0000,7.2260,4.6711,
-1.0000,8.1533,5.1771,
-1.0000,7.4825,6.2146,
-1.0000,7.0484,5.4555,
-1.0000,8.5084,5.9868,
-1.0000,7.5417,4.0891,
-1.0000,7.2063,2.3937,
-1.0000,6.5355,1.3310,
-1.0000,5.4503,1.7358,
-1.0000,5.8449,2.4443,
-1.0000,4.8979,3.1781,
-1.0000,5.8055,4.6711,
-1.0000,7.3641,5.9868,
-1.0000,6.2592,4.6711,
-1.0000,8.3703,7.5810,
-1.0000,8.5676,4.6457,
-1.0000,8.1676,4.6457,
)]

data= np.array(raw_data).reshape(100, 3)

figura = plt.figure()
ax = figura.add_subplot(111)
ax.scatter(data[:, 1], data[:, 2], c=data[:, 0])

Y = data[:, 0]
X = data[:, 1:]

# INICIALIZAR   ###########################################################
num_of_epochs = 40 #10000
lr = 1e-3
C = 30
# Add column vector of ones for computational convenience
X = np.column_stack((np.ones(len(X)), X))
# Initialize normal vector
w = np.ones(len(X[0]))
custo=0

# ###################modelo.fit()


for i in range(num_of_epochs+1):

    if i % 10 == 0:
        print(i, ' | ', custo)
        plota_corte(w, i)
    # calcula as distancias da atual regra de decisão (em função do w)
    distancias = Y * (np.dot(X, w)) - 1
    # if distance is more than 0, sample is not on the support vector -> Lagrange multiplier will be 0
    distancias[distancias > 0] = 0

    # calcula os custos da atual regra de decisão (em função do w)
    custo = 1 / 2 * np.dot(w, w) - C * np.sum(distancias)

    # calcula o gradiente para a função objetivo
    dw = np.zeros(len(w))
    for chave, valor in enumerate(distancias):
        if valor == 0:  # if sample is not on the support vector
            di = w  # (alpha * y[chave] * X[chave]) = 0
        else:
            # (alpha * y[chave] * X[chave]) = y[chave] * X[chave]
            di = w - (C * Y[chave] * X[chave])
        dw += di

    dw_medio = dw / len(X)

    w = w - lr * dw_medio

print(w)

modelo_temp = svm.SVC(kernel='linear').fit(data[:, 1:], data[:, 0])

x1_surf = np.arange(1, 9, 0.1)
x2_surf = -(modelo_temp.intercept_ + modelo_temp.coef_[:, 0]*x1_surf)/modelo_temp.coef_[:, 1]

ax.plot(x1_surf, x2_surf, linewidth=5, linestyle='dashed', color='black', label='SVM')
# ax.scatter(data[:, 1], data[:, 2], c=data[:, 0])

plt.legend()
plt.show()

# #########################################################################
'''
modelo_temp = svm.SVC(kernel='linear').fit(X, Y)

x1_surf = np.arange(0, 10, 0.1)
x2_surf = -(modelo_temp.intercept_ + modelo_temp.coef_[:, 0]*x1_surf)/modelo_temp.coef_[:, 1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x1_surf, x2_surf)
ax.scatter(data[:, 1], data[:, 2], c=data[:, 0])
plt.show()
'''