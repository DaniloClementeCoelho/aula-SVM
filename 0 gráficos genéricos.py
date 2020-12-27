import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

limite = 100
a, b = np.meshgrid(np.linspace(-limite, limite, 20*limite+1),
                   np.linspace(-limite, limite, 20*limite+1))
x1 = a.ravel()
x2 = b.ravel()

# função a ser estudada
beta0 = 1
beta1 = 1
beta2 = 1
beta12 = 1
beta11 = 0
beta22 = 0
Y = np.around(beta0 + beta1*x1 + beta2*x2 + beta12*x1*x2 + beta11*x1**2 + beta22*x2**2)

corte = -10

# cria target q é a variável a ser classificada
target = Y
for i in range(len(Y)):
    if Y[i] > corte:
        target[i] = 1
    else:
        target[i] = 0

# cria uma base paramodelo
base = pd.DataFrame({'x1': x1, 'x2': x2, 'Y': Y, 'target': target})
base_plot = base.sample(10000)

# cria base do plano cortado
plano_cortado = base[base['Y'] == corte]

# ############################# GRAFICO 3D Com corte threshold ##############################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(base_plot['x1'], base_plot['x2'], base_plot['Y'], c='blue', marker='.', s=1, alpha=0.1, label='superficie')
ax.scatter(plano_cortado['x1'], plano_cortado['x2'], plano_cortado['Y'],
           c='red', marker='o', s=2, alpha=0.9, label='hiperplano')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1,x2)')
ax.legend()
plt.show()


# ############################# GRAFICO 2D do hiperplano  ##############################
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(plano_cortado['x1'], plano_cortado['x2'], marker='o')
ax2.contourf(a, b, target.reshape(a.shape), cmap=plt.cm.coolwarm, alpha=0.8)
# ax.xlabel('x1')
# ax.ylabel('x2')
# ax.gca().set_aspect('equal')

plt.show()
