# rodar pelo console interativo do Python (Alt+Shift+E)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from funções.Gera_Base import gera_base
from funções.analise_inicial_superficie_gráfica import gera_graficos

# DELIMITA ESPAÇO A SER ESTUDADO
limite_eixos = 100  # quanto maior, maior será o tamanho da amostra e processamento
x1_surf, x2_surf = np.meshgrid(np.arange(1, limite_eixos, 0.1),
                               np.arange(1, limite_eixos, 0.1))
x1 = x1_surf.ravel()
x2 = x2_surf.ravel()
X = np.concatenate([x1.reshape(x1.size, 1), x2.reshape(x1.size, 1)], axis=1)


# ##########    PLANO LINEAR COMO LOGITO      #############
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
logito = np.around(beta0 + beta1 * np.log(x1) + beta2 * np.log(x2) + beta12 * x1 * x2 + beta11 * x1 ** 2 + beta22 * x2 ** 2, 2)

# ###########   INSERI O HIPERPLANO DE CORTE E O RUÍDO #############################
print(logito.min(), logito.max()) #olhar os limites para colocar um "corte" que faça sentido

# definir o hiperplano de corte a ser utilizado e a variação do ruído na geração da amostra
corte = 8            # depois de olhar os limites da função, escolher um corte que faça um "desenho interessante"
ruido = 0.5     # desvio padrão do ruído:para que as regressões não acertem 100%.Escolher valores adequados olhando os gráficos abaixo


# ####################   GERA BASE, SUPERFICIE E HIPERPLANO    ####################################

db = gera_base(x1, x2, logito, corte=corte, ruido=ruido)
amostra = db.sample(1000)
print('taxa vermelho:', round(db.target.mean()*100,1), '%')

# gera_graficos(db, x1_surf, x2_surf, corte)  # olhar o gráfico e ajustar o corte e ruído, se necessário
figura = plt.figure()
ax_2D = figura.add_subplot(111)
ax_2D.scatter(amostra['x1'], amostra['x2'],
              c=amostra['cor'], marker=',', s=4, alpha=0.9)
intercecção = db[round(db['logito_gabarito'], 2) == corte]
ax_2D.scatter(round(intercecção['x1']), round(intercecção['x2']),c='black', marker='o', s=1)
plt.show()
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

        # Logistico Log
logist_log = LogisticRegression(penalty='none')
logist_log.fit(np.log(X), db.target)
# logist_quad.named_steps['logisticregression'].coef_
db['prob_prev_log_quad'] = logist_log.predict_proba(np.log(X))[:, 1]
prob_prev_log_quad_surf = np.array(db['prob_prev_log_quad']).reshape(x1_surf.shape)
# Gráfico
ax2 = fig.add_subplot(222)
ax2.title.set_text('LOGÍSTICA Log')
ax2.contourf(x1_surf, x2_surf, prob_prev_log_quad_surf, cmap=plt.cm.coolwarm) # plota as probabilidade estimadas
#ax2.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

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
# ax3.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

# GABARITO
ax4 = fig.add_subplot(224)
ax4.title.set_text('GABARITO')
ax4.contourf(x1_surf, x2_surf, np.array(db['target_gabarito']).reshape(x1_surf.shape),
            cmap=plt.cm.coolwarm, alpha=0.5)# plota gabarito
# ax4.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos

fig.tight_layout()
plt.show()