import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

limite_eixos = 100
a, b = np.meshgrid(np.arange(-limite_eixos, limite_eixos, 0.1),
                   np.arange(-limite_eixos, limite_eixos, 0.1))
x1 = a.ravel()
x2 = b.ravel()

# função a ser estudada
beta0 = 1
beta1 = 1
beta2 = 1
beta12 = 1
beta11 = 1
beta22 = 1
Y = np.around(beta0 + beta1*x1 + beta2*x2 + beta12*x1*x2 + beta11*x1**2 + beta22*x2**2)

#plano que define o target
corte = 10000

# cria target q é a variável a ser , dependendo da superfície
target = np.empty(len(x1), dtype=int)
target_real = np.empty(len(x1), dtype=int)
cor = np.empty(len(x1), dtype=object)
cor_real = np.empty(len(x1), dtype=object)
erro = Y.mean()/3 #valor da aletorização - ruído não explicado pelo modelo

#simula reposta aleatória
for i in range(len(Y)):
    if Y[i] + np.random.normal(0, erro) > corte:
        target[i] = 1
        cor[i] = 'red'
    else:
        target[i] = 0
        cor[i] = 'blue'
#calcula o valor real sem aleatoriedade para gráfico de contorno
for i in range(len(Y)):
    if Y[i] > corte:
        target_real[i] = 1
        cor_real[i]='red'
    else:
        target_real[i] = 0
        cor_real[i] = 'blue'

# cria uma base para modelo
base = pd.DataFrame({'x1': x1, 'x2': x2, 'Y': Y, 'target': target, 'cor': cor})
base_plot = base.sample(10000)
amostra = base.sample(500)


# cria base do plano cortado
pontos_plano_cortado = base[base['Y'] == corte]
hiperplano_corte = np.zeros(a.size).reshape(a.shape)+corte

# ############################# GRAFICO 3D Com corte threshold ##############################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(base_plot['x1'], base_plot['x2'], base_plot['Y'], c=base_plot['target'], cmap=plt.cm.coolwarm, marker='.', s=1, alpha=0.1, label='superficie')
ax.scatter(pontos_plano_cortado['x1'], pontos_plano_cortado['x2'], pontos_plano_cortado['Y'], c='black', marker='o', s=2, alpha=0.9, label='hiperplano')
ax.plot_surface(a, b, hiperplano_corte, alpha=0.6, color = 'gray') #plano do corte
ax.plot_surface(a, b, Y.reshape(a.shape), alpha=0.3, color='black') #curva toda
ax.scatter(amostra['x1'], amostra['x2'], amostra['Y'], c=amostra['cor'], marker='x', s=3, alpha=0.9)

ax.set_xlabel('x1')
ax.set_xlim(-limite_eixos, limite_eixos)
ax.set_ylabel('x2')
ax.set_ylim(-limite_eixos, limite_eixos)
ax.set_zlabel('f(x1,x2)')
# ax.legend()
plt.show()


# ############################# GRAFICO 2D do hiperplano  ##############################
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
# ax2.scatter(plano_cortado['x1'], plano_cortado['x2'], marker='o')
ax2.contourf(a, b, target_real.reshape(a.shape), cmap=plt.cm.coolwarm, alpha=0.5)
ax2.scatter(amostra['x1'], amostra['x2'], c=amostra['cor'], marker='x', s=3, alpha=0.9)
# ax.xlabel('x1')
# ax.ylabel('x2')
# ax.gca().set_aspect('equal')

plt.show()
