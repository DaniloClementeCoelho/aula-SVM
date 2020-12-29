# ##################################### RODADA 1 ##############################################
# rodar pelo console interativo do Python (Alt+Shift+E)
# ##############################    INCLUI BIBLIOTECAS      ###################################
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ##################################### RODADA 2 ##############################################
# ##############################    DELIMITA O ESPAÇO ESTUDADO      ###################################
limite_eixos = 100  # quanto maior, maior será o tamanho da amostra e processamento
x1_surf, x2_surf = np.meshgrid(np.arange(-limite_eixos, limite_eixos, 0.1),
                               np.arange(-limite_eixos, limite_eixos, 0.1))
x1 = x1_surf.ravel()
x2 = x2_surf.ravel()

# ##################################### RODADA 3 ##############################################
# ##############################    ESCOLHE FUNÇÕES LOGITO      ###################################
# SUPERFÍCIE LINEAR/QUADRÁTICA BIMODAL
beta = {'intercepto'  : '1',
                'x1'  : '1',
                'x2'  : '1',
                'x1x2': '1',
                'x1^2': '1',
                'x2^2': '1'
        }
beta0 = float(beta['intercepto'])
beta1 = float(beta['x1'])
beta2 = float(beta['x2'])
beta12 = float(beta['x1x2'])
beta11 = float(beta['x1^2'])
beta22 = float(beta['x2^2'])
logito = np.around(beta0 + beta1 * x1 + beta2 * x2 + beta12 * x1 * x2 + beta11 * x1 ** 2 + beta22 * x2 ** 2, 2)


# NORMAL MULTIVARIADA BIMODAL
from funções.gera_normais_multivariadas import multivariate_gaussian
# Mean vector and covariance matrix
mu1 = np.array([-10., -10.])
Sigma1 = np.array([[1., -0.5], [-0.5,  1.5]])

mu2 = np.array([10., 10.])
Sigma2 = np.array([[10., 0], [0,  10.]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(x1_surf.shape + (2,))
pos[:, :, 0] = x1_surf
pos[:, :, 1] = x2_surf

logito = multivariate_gaussian(pos, mu1, Sigma1) + multivariate_gaussian(pos, mu2, Sigma2)

# ##################################### RODADA 4 ##############################################
# print(logito.min(), logito.max()) #olhar os limites para colocar um "corte" que faça sentido

# definir o hiperplano de corte a ser utilizado e a variação do ruído na geração da amostra
corte = 10000            # depois de olhar os limites da função, escolher um corte que faça um "desenho interessante"
ruido = abs(corte)/3     # desvio padrão do ruído:para que as regressões não acertem 100%.
                         # Escolher valores adequados olhando os gráficos abaixo

# ##################################### RODADA 5 ##############################################
# ####################   GERA BASE, SUPERFICIE E HIPERPLANO    ####################################
from funções.Gera_Base import gera_base
db = gera_base(x1, x2, logito, corte=corte, ruido=ruido)

from funções.analise_superficie_gráfica import gera_graficos
gera_graficos(db, x1_surf, x2_surf, corte)  # olhar o gráfico e ajustar o corte e ruído, se necessário

# ##################################### RODADA 6 ##############################################
# ##############################   CONFIGURA GRÁFICOS    ######################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('x1')
ax.set_xlim(-limite_eixos, limite_eixos)
ax.set_ylabel('x2')
ax.set_ylim(-limite_eixos, limite_eixos)


# ##################################### RODADA 7 ##############################################
# ###################    AJUSTA OS MODELOS   ######################################
from funções.maxima_acuracia import max_acuracia
from funções.superficie_separacao_maxima import sup_sep_max
amostra = db.sample(1000)


ax.scatter(amostra.x1, amostra.x2, c=amostra.cor)

# MODELO LINEAR SIMPLES
logistica1 = smf.logit(formula='target ~ x1 + x2', data=amostra).fit()
logistica1.summary()
max_acu1, vec_acu1, logito_otimo_ajustado1 = max_acuracia(logistica1)
#calcula e plota superfície de separação máxima
superficie_otima1 = sup_sep_max(db, logistica1, logito_otimo_ajustado1)
ax.scatter(superficie_otima1.x1, superficie_otima1.x2, c='black', marker=',', s=1, label=max_acu1)

# MODELO LINEAR QUADRÁTICO

logistica2 = smf.logit(formula='target ~ x1 + x2 + I(x1*x2) + I(x1*x1) + I(x2*x2)', data=amostra).fit()
logistica2.summary()
max_acu2, vec_acu2, logito_otimo_ajustado2 = max_acuracia(logistica2)
# calcula e plota superfície de separação máxima
superficie_otima2 = sup_sep_max(db, logistica2, logito_otimo_ajustado2)
ax.scatter(superficie_otima2.x1, superficie_otima2.x2, c='black', marker=',', s=1, label=max_acu2)

plt.legend()

# plota gabarito
ax.contourf(x1_surf, x2_surf, np.array(db['target_gabarito']).reshape(x1_surf.shape),
            cmap=plt.cm.coolwarm, alpha=0.5)


# ############################# GRAFICOS 4D  ##############################
# superficie plotada com curva teórica pintada de acordo com a probabilidade prevista
fig = plt.figure()
ax = fig.gca(projection='3d')
logito_surf = np.array(db['logito_gabarito']).reshape(x1_surf.shape)
prob_prev_surf = np.array(db['prob_prev']).reshape(x1_surf.shape)
surf = ax.plot_surface(x1_surf, x2_surf, logito_surf, cmap=plt.cm.coolwarm)

# ###### CONFIGURA GRÁFICOS PRA IR MOSTRANDO PONTO A PONTO
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_xlabel('x1')
ax3d.set_xlim(-limite_eixos, limite_eixos)
ax3d.set_ylabel('x2')
ax3d.set_ylim(-limite_eixos, limite_eixos)
ax3d.set_zlabel('f(x1,x2)')
# superficie plotada com curva teórica preta opaca
logito_surf = np.array(db['logito_gabarito']).reshape(x1_surf.shape)
ax3d.plot_surface(x1_surf, x2_surf, logito_surf,
                  label='superficie', color='black', alpha=0.3)  # curva toda

# Todos os pontos da amostra no gráfico coloridos
ax3d.scatter(amostra['x1'], amostra['x2'], amostra['logito_gabarito'],
             c=amostra['cor'], marker='x', s=3, alpha=0.9)

# hiperplano escolhido para geração da amostra e intersecção com suérfície
hiperplano_corte = np.zeros(x1_surf.size).reshape(x1_surf.shape) + corte
ax3d.plot_surface(x1_surf, x2_surf, hiperplano_corte,
                  alpha=0.3, color='orange', label='hiperplano')  # plano do corte
base_pontos_plano_cortado = db[round(db['logito_gabarito']) == corte]
ax3d.scatter(base_pontos_plano_cortado['x1'], base_pontos_plano_cortado['x2'], base_pontos_plano_cortado['logito_gabarito'],
             c='black', marker='o', s=2, alpha=0.9)

ax3d.legend()

'''
#superficie plotada ponto a ponto e colorida com target
base_plot = db.sample(1000)
ax3d.scatter(base_plot['x1'], base_plot['x2'], base_plot['logito_gabarito'],
             c=base_plot['target'], cmap=plt.cm.coolwarm, marker='.', s=1, alpha=0.4)
'''