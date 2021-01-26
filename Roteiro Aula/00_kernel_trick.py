# rodar pelo console interativo do Python (Alt+Shift+E)
import numpy as np
from funções.Gera_Base import gera_base
import matplotlib.pyplot as plt

# DELIMITA ESPAÇO A SER ESTUDADO
limite_eixos = 50  # quanto maior, maior será o tamanho da amostra e processamento
x1_surf, x2_surf = np.meshgrid(np.arange(-limite_eixos, limite_eixos, 0.1),
                               np.arange(-limite_eixos, limite_eixos, 0.1))
x1 = x1_surf.ravel()
x2 = x2_surf.ravel()
X = np.concatenate([x1.reshape(x1.size, 1), x2.reshape(x1.size, 1)], axis=1)


# ##########    PLANO LINEAR COMO LOGITO      #############
beta = {'intercepto'  : '1',
                'x1'  : '1',
                'x2'  : '1',
                'x1x2': '1',
                'x1^2': '1',
                'x2^2': '1'}
beta0 = float(beta['intercepto'])
beta1 = float(beta['x1'])
beta2 = float(beta['x2'])
beta12 = float(beta['x1x2'])
beta11 = float(beta['x1^2'])
beta22 = float(beta['x2^2'])
logito = np.around(beta0 + beta1 * x1 + beta2 * x2 + beta12 * x1 * x2 + beta11 * x1 ** 2 + beta22 * x2 ** 2, 2)

# ###########   INSERI O HIPERPLANO DE CORTE E O RUÍDO #############################
print(logito.min(), logito.max()) #olhar os limites para colocar um "corte" que faça sentido

# definir o hiperplano de corte a ser utilizado e a variação do ruído na geração da amostra
corte = 1000            # depois de olhar os limites da função, escolher um corte que faça um "desenho interessante"
ruido = 200    # desvio padrão do ruído:para que as regressões não acertem 100%.Escolher valores adequados olhando os gráficos abaixo


# ####################   GERA BASE, SUPERFICIE E HIPERPLANO    ####################################

db = gera_base(x1, x2, logito, corte=corte, ruido=ruido)
base_plot = db.sample(1000)


figura = plt.figure()
ax= figura.add_subplot(111, projection='3d')
ax.view_init(elev=90., azim=0)
plt.tight_layout()
ax.scatter(base_plot.x1, base_plot.x2, base_plot.logito_gabarito, c=base_plot.cor)
plt.show()