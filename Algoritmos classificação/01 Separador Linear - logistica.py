import numpy as np
import math
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
#################################### CRIA A BASE ##########################################
n = 10000

X1 = np.random.normal(1, 1, n).reshape(n, 1)
X2 = np.random.normal(1, 1, n).reshape(n, 1)
logito = 2 + 1*X1 - 3*X2
X = np.concatenate([X1, X2], axis=1)

target = np.empty(n, dtype=float)
cor = np.empty(n, dtype=object)
prob_verde = np.empty(n, dtype=float)

for i in range(len(X1)):
    prob_verde[i] = math.exp(logito[i]) / (1 + math.exp(logito[i]))
    if np.random.rand(1) < prob_verde[i]:
        target[i] = 1
        cor[i] = 'g'
    else:
        target[i] = 0
        cor[i] = 'b'

#################################### AJUSTA MODELO ##########################################
modelo_logistico= LogisticRegression().fit(X, target)
prob_verde_prevista = modelo_logistico.predict_proba(X)[:,1]
logito_previsto=prob_verde_prevista
for i in range(len(prob_verde_prevista)):
    logito_previsto[i] = math.log(prob_verde_prevista[i]/(1-prob_verde_prevista[i]))

'''  

predito_log = modelo_logistico.predict_log_proba(X)
k=1
logito_calculado=math.log(predito[k,1]/(1-predito[k,1]))
print ( X1[k], X2[k], prob_verde_prevista[k], predito[k,1], predito_log[k,1], logito_calculado)

modelo_logistico.intercept_ + np.dot(modelo_logistico.coef_ , X[k].reshape(2, 1))

modelo_logistico.classes_
modelo_logistico.coef_
modelo_logistico.intercept_
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
ax.plot(X1, (2 + 1*X1)/3, c='black')
plt.show()

#Gráficos de dispersão separados com curva ajustada
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs[0].scatter(X1[target==0], X2[target==0])
axs[0].plot(X1, (2 + 1*X1)/3, c='black')
axs[1].scatter(X1[target==1], X2[target==1])
axs[1].plot(X1, (2 + 1*X1)/3, c='black')
plt.show()


#################################### MONTA GRÁFICOS SUPERFICIE##########################################
x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 10),
                               np.linspace(X2.min(), X2.max(), 10)
                               )
X_surf = np.concatenate([x1_surf.reshape(100,1), x2_surf.reshape(100,1)], axis=1)
prob_verde_surf=modelo_logistico.predict_proba(X_surf)[:,1]
logito_surf=prob_verde_surf
for i in range(len(prob_verde_surf)):
    logito_surf[i] = math.log(prob_verde_surf[i]/(1-prob_verde_surf[i]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_surf, x2_surf, logito_surf.reshape(x1_surf.shape), color='red', alpha=0.5)
ax.scatter(X1, X2, c=cor)
plt.show()