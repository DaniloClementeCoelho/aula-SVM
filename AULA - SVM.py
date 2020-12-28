# rodar pelo console interativo do Python

# ##############################    INCLUI BIBLIOTECAS      ###################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import math


# ##############################        GERA BASE          ####################################
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

# ##############################        GRÁFICOS          ####################################
#scatter de x1 e x2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('x1')
ax.set_xlim(-limite_eixos, limite_eixos)
ax.set_ylabel('x2')
ax.set_ylim(-limite_eixos, limite_eixos)

# 2 variaveis independentes
ax.scatter(amostra.x1, amostra.x2)

# gráfico 3D com a cor representando o target
ax.scatter(amostra.x1, amostra.x2, c=amostra.cor)

# ajuste do modelo linear

def max_acuracia(modelo):
    acc = np.empty(100, dtype=float)
    max_acc = 0
    for i in range(0, 100, 1):
        acc[i] = (modelo.pred_table(threshold=i/100)[0,0]+modelo.pred_table(threshold=i/100)[1,1])/modelo.pred_table(threshold=i/100).sum()
        max_acc = max(max_acc, acc[i])
        if max_acc == acc[i]:
            corte = i/100
    return max_acc, corte

logistica1 = smf.logit(formula='target ~ x1 + x2', data=amostra).fit()
# dir(logistica1)
logistica1.summary()
accuracy1, prob_corte1 = acuracia(logistica1)
print(accuracy1, prob_corte1)
logito_corte1= round(math.log( (prob_corte1)/(1-prob_corte1)), 2)

# ajuste do modelo completo
logistica2 = smf.logit(formula='target ~ x1 + x2 + I(x1*x2) + I(x1*x1) + I(x2*x2)', data=amostra).fit()
logistica2.summary()
accuracy2, prob_corte2 = acuracia(logistica2)
print(accuracy2, prob_corte2)
logito_corte2= round(math.log( (prob_corte2)/(1-prob_corte2)), 2)


#calcula superfície de separação máxima modelo 1
base['prob_prev'] = logistica1.predict(base)
base['logito_prev'] = round( np.log( (base['prob_prev'])/(1-(base['prob_prev'])) ),2)
superficie_otima = base[base['logito_prev']==logito_corte1]
print(base['logito_prev'].min(), base['logito_prev'].max(),logito_corte1)
superficie_otima.shape
# plota o ajuste do modelo 1
ax.scatter(superficie_otima.x1, superficie_otima.x2,
           c='black', marker=',', s=1)


#calcula superfície de máxima separação modelo 2
base['prob_prev'] = logistica2.predict(base)
base['logito_prev'] = round( np.log( (base['prob_prev'])/(1-(base['prob_prev'])) ),2)
superficie_otima = base[base['logito_prev']==logito_corte2]
print(base['logito_prev'].min(), base['logito_prev'].max(),logito_corte2)
superficie_otima.shape
# plota o ajuste do modelo 2
ax.scatter(superficie_otima.x1, superficie_otima.x2,
           c='black', marker=',', s=1)


# plota gabarito
ax.contourf(a, b, target_real.reshape(a.shape), cmap=plt.cm.coolwarm, alpha=0.5)


