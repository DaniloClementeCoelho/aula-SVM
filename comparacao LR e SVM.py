# rodar pelo console interativo do Python (Alt+Shift+E)
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve, classification_report,\
                            accuracy_score, confusion_matrix, auc

from funções.gera_normais_multivariadas import multivariate_gaussian
from funções.Gera_Base import gera_base
from funções.analise_superficie_gráfica import gera_graficos

# ##################################### RODADA 1 ##############################################
# ######################    DELIMITA O ESPAÇO ESTUDADO      ###################################
limite_eixos = 50  # quanto maior, maior será o tamanho da amostra e processamento
x1_surf, x2_surf = np.meshgrid(np.arange(-limite_eixos, limite_eixos, 0.1),
                               np.arange(-limite_eixos, limite_eixos, 0.1))
x1 = x1_surf.ravel()
x2 = x2_surf.ravel()


# ##################################### RODADA 2 ##############################################
# ##########################    ESCOLHE FUNÇÕES LOGITO      ###################################
# NORMAL MULTIVARIADA BIMODAL

# Mean vector and covariance matrix
mu1 = np.array([-30., -20.])
Sigma1 = np.array([[50., -0.9], [-0.9,  30]])

mu2 = np.array([30., 30.])
Sigma2 = np.array([[30., 10], [10,  30.]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(x1_surf.shape + (2,))
pos[:, :, 0] = x1_surf
pos[:, :, 1] = x2_surf

logito = 1000*(multivariate_gaussian(pos, mu1, Sigma1) + multivariate_gaussian(pos, mu2, Sigma2)).ravel()

# ##################################### RODADA 3 ##############################################
print(logito.min(), logito.max()) #olhar os limites para colocar um "corte" que faça sentido

# definir o hiperplano de corte a ser utilizado e a variação do ruído na geração da amostra
corte = 1            # depois de olhar os limites da função, escolher um corte que faça um "desenho interessante"
ruido = 0.5     # desvio padrão do ruído:para que as regressões não acertem 100%.
                         # Escolher valores adequados olhando os gráficos abaixo

# ##################################### RODADA 4 ##############################################
# ####################   GERA BASE, SUPERFICIE E HIPERPLANO    ####################################

db = gera_base(x1, x2, logito, corte=corte, ruido=ruido)
print('taxa vermelho:', round(db.target.mean()*100,1), '%')

gera_graficos(db, x1_surf, x2_surf, corte)  # olhar o gráfico e ajustar o corte e ruído, se necessário

# ###############################    AJUSTA OS MODELOS   ######################################

X = np.concatenate([x1.reshape(x1.size, 1), x2.reshape(x1.size, 1)], axis=1)
pipe = make_pipeline(PolynomialFeatures(2), LogisticRegression(penalty='none'))
logistica = pipe.fit(X, db.target)

db['prob_prev'] = logistica.predict_proba(X)[:, 1]
prob_prev_surf = np.array(db['prob_prev']).reshape(x1_surf.shape)
amostra = db.sample(1000)

fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
'''ax.set_xlabel('x1')
ax.set_xlim(-limite_eixos, limite_eixos)
ax.set_ylabel('x2')
ax.set_ylim(-limite_eixos, limite_eixos)
ax2.set_xlabel('x1')
ax2.set_xlim(-limite_eixos, limite_eixos)
ax2.set_ylabel('x2')
ax2.set_ylim(-limite_eixos, limite_eixos)'''

ax.contourf(x1_surf, x2_surf, prob_prev_surf, cmap=plt.cm.coolwarm) # plota as probabilidade estimadas
ax.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

ax2.contourf(x1_surf, x2_surf, np.array(db['target_gabarito']).reshape(x1_surf.shape),
            cmap=plt.cm.coolwarm, alpha=0.5)# plota gabarito
ax2.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

fig.tight_layout()