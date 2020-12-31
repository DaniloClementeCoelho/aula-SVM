# rodar pelo console interativo do Python (Alt+Shift+E)
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report,\
                            accuracy_score, confusion_matrix, auc

from funções.gera_normais_multivariadas import multivariate_gaussian
from funções.Gera_Base import gera_base
from funções.analise_inicial_superficie_gráfica import gera_graficos

# DELIMITA ESPAÇO A SER ESTUDADO
limite_eixos = 50  # quanto maior, maior será o tamanho da amostra e processamento
x1_surf, x2_surf = np.meshgrid(np.arange(-limite_eixos, limite_eixos, 1),
                               np.arange(-limite_eixos, limite_eixos, 1))
x1 = x1_surf.ravel()
x2 = x2_surf.ravel()
X = np.concatenate([x1.reshape(x1.size, 1), x2.reshape(x1.size, 1)], axis=1)


# ##########    NORMAL MULTIVARIADA BIMODAL COMO LOGITO      ##################################
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

# ###########   INSERI O HIPERPLANO DE CORTE E O RUÍDO #############################
print(logito.min(), logito.max()) #olhar os limites para colocar um "corte" que faça sentido

# definir o hiperplano de corte a ser utilizado e a variação do ruído na geração da amostra
corte = 1            # depois de olhar os limites da função, escolher um corte que faça um "desenho interessante"
ruido = 0.5     # desvio padrão do ruído:para que as regressões não acertem 100%.Escolher valores adequados olhando os gráficos abaixo


# ####################   GERA BASE, SUPERFICIE E HIPERPLANO    ####################################

db = gera_base(x1, x2, logito, corte=corte, ruido=ruido)
amostra = db.sample(1000)
print('taxa vermelho:', round(db.target.mean()*100,1), '%')

gera_graficos(db, x1_surf, x2_surf, corte)  # olhar o gráfico e ajustar o corte e ruído, se necessário

# ###############################    AJUSTA OS MODELOS   ######################################
fig = plt.figure()
        # Logistico linear
logist_linear = LogisticRegression(penalty='none')
logist_linear.fit(X, db.target)
db['prob_prev_log_linear'] = logist_linear.predict_proba(X)[:, 1]
prob_prev_log_lin_surf = np.array(db['prob_prev_log_linear']).reshape(x1_surf.shape)
# Gráfico
ax1 = fig.add_subplot(221)
ax1.title.set_text('LOGÍSTICA')
ax1.contourf(x1_surf, x2_surf, prob_prev_log_lin_surf, cmap=plt.cm.coolwarm) # plota as probabilidade estimadas
# ax1.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

        # Logistico Quadrático
logist_quad = make_pipeline(PolynomialFeatures(2), LogisticRegression(penalty='none'))
logist_quad.fit(X, db.target)
# logist_quad.named_steps['logisticregression'].coef_
db['prob_prev_log_quad'] = logist_quad.predict_proba(X)[:, 1]
prob_prev_log_quad_surf = np.array(db['prob_prev_log_quad']).reshape(x1_surf.shape)
# Gráfico
ax2 = fig.add_subplot(222)
ax2.title.set_text('LOGÍSTICA QUADRÁTICA')
ax2.contourf(x1_surf, x2_surf, prob_prev_log_quad_surf, cmap=plt.cm.coolwarm) # plota as probabilidade estimadas
# ax2.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

        # SVM
SVM = svm.SVC()
SVM.fit(X, db.target)
db['prev_SVM'] = SVM.predict(X)
prob_prev_SVM_surf = np.array(db['prev_SVM']).reshape(x1_surf.shape)
# print("Acurácia:", accuracy_score(y_test, y_pred))# Gráfico
ax3 = fig.add_subplot(223)
ax3.title.set_text('SVM')
ax3.contourf(x1_surf, x2_surf, prob_prev_SVM_surf, cmap=plt.cm.coolwarm) # plota as probabilidade estimadas
# ax3.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

# GABARITO
ax4 = fig.add_subplot(224)
ax4.title.set_text('GABARITO')
ax4.contourf(x1_surf, x2_surf, np.array(db['target_gabarito']).reshape(x1_surf.shape),
            cmap=plt.cm.coolwarm, alpha=0.5)# plota gabarito
# ax4.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

fig.tight_layout()
plt.show()