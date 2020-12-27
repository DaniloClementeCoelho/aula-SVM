import numpy as np
import math
#################################### CRIA A BASE ##########################################
n = 10000

X1 = np.random.normal(1, 1, n).reshape(n, 1)
X2 = np.random.normal(1, 1, n).reshape(n, 1)

X = np.concatenate([X1, X2], axis=1)

target = np.empty(n, dtype=float)
cor = np.empty(n, dtype=object)

erro = 0.7

for i in range(len(X1)):
    if (X1[i]-1)**2 + (X2[i]-1)**2 + np.random.normal(0, erro) > 4:
        target[i] = 1
        cor[i] = 'g'
    else:
        target[i] = 0
        cor[i] = 'b'


#################################### AJUSTA MODELO ##########################################
from sklearn.linear_model import LogisticRegression

modelo_logistico= LogisticRegression().fit(X, target)
prob_verde_prevista = modelo_logistico.predict_proba(X)[:,1]

'''  

predito_log = modelo_logistico.predict_log_proba(X)
k=1
logito_calculado=math.log(predito[k,1]/(1-predito[k,1]))
print ( X1[k], X2[k], prob_verde_prevista[k], predito[k,1], predito_log[k,1], logito_calculado)

modelo_logistico.intercept_ + np.dot(modelo_logistico.coef_ , X[k].reshape(2, 1))

modelo_logistico.classes_
print(modelo_logistico.intercept_,modelo_logistico.coef_)
p1= modelo_logistico.coef_[0,0]
p2= modelo_logistico.coef_[0,1]
modelo_logistico.coef_.[1,1]
modelo_logistico.score(X, target)
'''
#################################### MONTA GRÁFICOS ##########################################
import matplotlib.pyplot as plt

#Gráficos de dispersão separados
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs[0].scatter(X1[target==0], X2[target==0])
axs[1].scatter(X1[target==1], X2[target==1])
plt.show()

#Gráfico de dispersão pintados
fig, ax = plt.subplots()
ax.scatter(X1, X2, c=cor)
plt.show()

#Gráfico de dispersão 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, target,c=cor)
plt.show()


#Gráfico de dispersão 2D com a curva da logística
fig, ax = plt.subplots()
ax.scatter(X1, X2, c=cor)
ax.plot(X1, (modelo_logistico.intercept_ + modelo_logistico.coef_[0,0]*X1)/modelo_logistico.coef_[0,1]*(-1), c='red')
plt.show()

#Gráficos de dispersão separados com curva ajustada
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs[0].scatter(X1[target==0], X2[target==0])
axs[0].plot(X1, (modelo_logistico.intercept_ + modelo_logistico.coef_[0,0]*X1)/modelo_logistico.coef_[0,1]*(-1), c='red')
axs[1].scatter(X1[target==1], X2[target==1])
axs[1].plot(X1, (modelo_logistico.intercept_ + modelo_logistico.coef_[0,0]*X1)/modelo_logistico.coef_[0,1]*(-1), c='red')
plt.show()



############################INSERIR TERMOS QUADRÁTICOS NA LOGISTICA ##########################################
X_quad = np.concatenate([X, X1**2, X2**2], axis=1)

modelo_logistico2= LogisticRegression().fit(X_quad, target)
#modelo_logistico.score(X, target)
modelo_logistico2.score(X_quad, target)

modelo_logistico2.coef_
modelo_logistico2.intercept_


x1_surf, x2_surf = np.meshgrid(
                                np.linspace(-10, 10, 10),
                                np.linspace(-10, 10, 10)
                             )
X_surf= np.concatenate([x1_surf, x2_surf], axis=1)
pred_surf=modelo_logistico2.predict_proba(X_surf)[:,1]
logito_surf = math.log(pred_surf/(1-pred_surf))



#Gráfico de dispersão 2D com a curva da logística
fig, ax = plt.subplots()
ax.scatter(X1, X2, c=cor)
ax.plot(X1, (, c='red')
plt.show()

#Gráficos de dispersão separados com curva ajustada
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs[0].scatter(X1[target==0], X2[target==0])
axs[0].plot(X1, (modelo_logistico2.intercept_ + modelo_logistico2.coef_[0,0]*X1)/modelo_logistico2.coef_[0,1]*(-1), c='red')
axs[1].scatter(X1[target==1], X2[target==1])
axs[1].plot(X1, (modelo_logistico2.intercept_ + modelo_logistico2.coef_[0,0]*X1)/modelo_logistico2.coef_[0,1]*(-1), c='red')
plt.show()
