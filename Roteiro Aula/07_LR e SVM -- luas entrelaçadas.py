# rodar pelo console interativo do Python (Alt+Shift+E)
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons

# DELIMITA ESPAÇO A SER ESTUDADO
limite_eixos = 3  # quanto maior, maior será o tamanho da amostra e processamento
x1_surf, x2_surf = np.meshgrid(np.arange(-limite_eixos, limite_eixos, 0.1),
                               np.arange(-limite_eixos, limite_eixos, 0.1))
x1 = x1_surf.ravel()
x2 = x2_surf.ravel()
X = np.concatenate([x1.reshape(x1.size, 1), x2.reshape(x1.size, 1)], axis=1)


# ##########    GERA LUAS      #############
Xa, ya = make_moons(n_samples=len(x1), noise=0.1)
db = DataFrame(dict(x1=Xa[:, 0], x2=Xa[:, 1], target=ya, cor='blue'))
cor = {0:'blue', 1:'red'}
for i in range(len(Xa)):
    db.iloc[i, 3] = cor[db.target[i]]

amostra = db.sample(1000)
print('taxa vermelho:', round(db.target.mean()*100,1), '%')

# gera_graficos(db, x1_surf, x2_surf, corte)  # olhar o gráfico e ajustar o corte e ruído, se necessário
figura = plt.figure()
ax_2D = figura.add_subplot(111)
ax_2D.scatter(amostra['x1'], amostra['x2'],
              c=amostra['cor'], marker=',', s=4, alpha=0.9)
# intercecção = db[round(db['logito_gabarito']) == corte]
# ax_2D.scatter(round(intercecção['x1']), round(intercecção['x2']), c='black', marker='o', s=1)
plt.gca().set_aspect('equal')
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
ax1.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos
plt.gca().set_aspect('equal')

        # Logistico Quadrático
logist_quad = make_pipeline(PolynomialFeatures(4), LogisticRegression(penalty='none'))
logist_quad.fit(X, db.target)
# logist_quad.named_steps['logisticregression'].coef_
db['prob_prev_log_quad'] = logist_quad.predict_proba(X)[:, 1]
prob_prev_log_quad_surf = np.array(db['prob_prev_log_quad']).reshape(x1_surf.shape)
# Gráfico
ax2 = fig.add_subplot(222)
ax2.title.set_text('LOGÍSTICA ordem 3')
ax2.contourf(x1_surf, x2_surf, prob_prev_log_quad_surf, cmap=plt.cm.coolwarm) # plota as probabilidade estimadas
ax2.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos
plt.gca().set_aspect('equal')

        # SVM
SVM = svm.SVC()
sample = db.sample(1500)
X_sample = np.concatenate([np.array(sample.x1).reshape(sample.x1.size, 1),
                           np.array(sample.x2).reshape(sample.x2.size, 1)], axis=1)
SVM.fit(X_sample, sample.target)
db['prev_SVM'] = SVM.predict(X)
prob_prev_SVM_surf = np.array(db['prev_SVM']).reshape(x1_surf.shape)
# print("Acurácia:", accuracy_score(y_test, y_pred))# Gráfico
ax3 = fig.add_subplot(223)
ax3.title.set_text('SVM')
ax3.contourf(x1_surf, x2_surf, prob_prev_SVM_surf, cmap=plt.cm.coolwarm, alpha=0.5) # plota as probabilidade estimadas
ax3.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.9, s=2) #plota os pontos
plt.gca().set_aspect('equal')

fig.tight_layout()
plt.show()