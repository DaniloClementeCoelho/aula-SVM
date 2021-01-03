
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import svm
import statsmodels.formula.api as smf
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


# ##########    NORMAL SUPERFICIE COMO LOGITO      ############# Mean vector and covariance matrix
beta = {'intercepto'  : '1',
                'x1'  : '1',
                'x2'  : '1',
                'x1x2': '0',
                'x1^2': '0',
                'x2^2': '0'}
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
corte = 20            # depois de olhar os limites da função, escolher um corte que faça um "desenho interessante"
ruido = 20     # desvio padrão do ruído:para que as regressões não acertem 100%.Escolher valores adequados olhando os gráficos abaixo


# ####################   GERA BASE, SUPERFICIE E HIPERPLANO    ####################################

db = gera_base(x1, x2, logito, corte=corte, ruido=ruido)
amostra = db.sample(1000)
print('taxa vermelho:', round(db.target.mean()*100,1), '%')

gera_graficos(db, x1_surf, x2_surf, corte)  # olhar o gráfico e ajustar o corte e ruído, se necessário

# ###############################    AJUSTA OS MODELOS   ######################################
        # Logistico -- escalas iguais
logistica1 = smf.logit('target ~ x1 + x2', data=db).fit()
params_log1 =  np.array(round(logistica1.params, 5)).reshape(3,1)

        # Logistico -- escalas diferentes
db['x2_2'] = 5000 + db.x2*1000
# db.x2_2.describe()      db.x2.describe()
params_teorico = np.array([np.round(params_log1[0] - 5000*params_log1[2]/1000, 5),
                           np.round(params_log1[1], 5),
                           np.round(params_log1[2]/1000, 5)])

logistica2 = smf.logit('target ~ x1 + x2_2', data=db).fit()
params_log2 = np.array(round(logistica2.params, 5))
print(np.concatenate([params_teorico.reshape(3, 1), params_log2.reshape(3, 1)], axis=1))









        # Logistico -- escalas diferentes
logistica1 = smf.logit('target ~ x1 + x2', data=db).fit()
params_log1 = np.exp(logistica1.params)


logist1.fit(X, db.target)
db['prob_prev_logist1'] = logist1.predict_proba(X)[:, 1]
prob_prev_log_lin_surf = np.array(db['prob_prev_logist1']).reshape(x1_surf.shape)
logist_linear.intercept_
logist_linear.coef_

# Gráfico
ax1 = fig.add_subplot(221)
ax1.title.set_text('LOGÍSTICA')
ax1.contourf(x1_surf, x2_surf, prob_prev_log_lin_surf, cmap=plt.cm.coolwarm) # plota as probabilidade estimadas
#ax1.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

        # SVM
SVM = svm.SVC()
sample = db.sample(10000)
X_sample = np.concatenate([np.array(sample.x1).reshape(sample.x1.size, 1),
                           np.array(sample.x2).reshape(sample.x2.size, 1)], axis=1)
SVM.fit(X_sample, sample.target)
db['prev_SVM'] = SVM.predict(X)
prob_prev_SVM_surf = np.array(db['prev_SVM']).reshape(x1_surf.shape)
# print("Acurácia:", accuracy_score(y_test, y_pred))# Gráfico
ax3 = fig.add_subplot(223)
ax3.title.set_text('SVM')
ax3.contourf(x1_surf, x2_surf, prob_prev_SVM_surf, cmap=plt.cm.coolwarm) # plota as probabilidade estimadas




# GABARITO
ax4 = fig.add_subplot(224)
ax4.title.set_text('GABARITO')
ax4.contourf(x1_surf, x2_surf, np.array(db['target_gabarito']).reshape(x1_surf.shape),
            cmap=plt.cm.coolwarm, alpha=0.5)# plota gabarito
# ax4.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

fig.tight_layout()
plt.show()