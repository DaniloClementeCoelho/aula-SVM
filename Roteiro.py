# rodar pelo console interativo do Python (Alt+Shift+E)
# ##################################### RODADA 1 ##############################################
# ##############################    DELIMITA O ESPAÇO ESTUDADO      ###################################
import numpy as np
limite_eixos = 50  # quanto maior, maior será o tamanho da amostra e processamento
x1_surf, x2_surf = np.meshgrid(np.arange(-limite_eixos, limite_eixos, 0.1),
                               np.arange(-limite_eixos, limite_eixos, 0.1))
x1 = x1_surf.ravel()
x2 = x2_surf.ravel()

# ##################################### RODADA 2 ##############################################
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
from funções.Gera_Base import gera_base
db = gera_base(x1, x2, logito, corte=corte, ruido=ruido)
print('taxa vermelho:', round(db.target.mean()*100,1), '%')
from funções.analise_superficie_gráfica import gera_graficos
gera_graficos(db, x1_surf, x2_surf, corte)  # olhar o gráfico e ajustar o corte e ruído, se necessário

# ##################################### RODADA 5 ##############################################
# ##############################   CONFIGURA GRÁFICOS    ######################################
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('x1')
ax.set_xlim(-limite_eixos, limite_eixos)
ax.set_ylabel('x2')
ax.set_ylim(-limite_eixos, limite_eixos)


# ##################################### RODADA 6 ##############################################
# ###################    AJUSTA OS MODELOS   ######################################
import statsmodels.formula.api as smf
from funções.indicadores_performace_otimos import max_acuracia
# from funções.indicadores_performace_otimos import max_sensibilidade
from funções.superficie_separacao_maxima import sup_sep_max
amostra = db.sample(1000)

# MODELO LINEAR SIMPLES
logistica1 = smf.logit(formula='target ~ x1 + x2', data=amostra).fit()
logistica1.summary()
max_acu1, prob_max1, vec_acu1, logito_otimo_ajustado1 = max_acuracia(logistica1)
print('   maxima acuracia:', round(max_acu1*100,1), '%',
      '   prob_acuracia_max:', prob_max1,
      '   tx_azul', round((1-db.target.mean())*100,1), '%')#calcula e plota superfície de separação máxima
superficie_otima1 = sup_sep_max(db, logistica1, logito_otimo_ajustado1)
print(db['prob_prev'].min(), db['prob_prev'].max(), prob_max1)
ax.scatter(superficie_otima1.x1, superficie_otima1.x2,
           c='black', marker=',', s=1, label=max_acu1)

# MODELO LINEAR QUADRÁTICO

logistica2 = smf.logit(formula='target ~ x1 + x2 + I(x1*x2) + I(x1*x1) + I(x2*x2)', data=amostra).fit()
logistica2.summary()
max_acu2, prob_max2, vec_acu2, logito_otimo_ajustado2 = max_acuracia(logistica2)
# max_sen2, prob_max2, vec_sen2, logito_otimo_ajustado2 = max_sensibilidade(logistica2)
print('   maxima acuracia:', round(max_acu2*100,1), '%',
      '   prob_acuracia_max:', prob_max2,
      '   tx_azul', round((1-db.target.mean())*100,1), '%')
# calcula e plota superfície de separação máxima
superficie_otima2 = sup_sep_max(db, logistica2, logito_otimo_ajustado2)
print(db['prob_prev'].min(), db['prob_prev'].max(), prob_max2)
ax.scatter(superficie_otima2.x1, superficie_otima2.x2,
           c='black', marker=',', s=1, label=max_acu2)

plt.legend()




ax.contourf(x1_surf, x2_surf, prob_prev_surf, cmap=plt.cm.coolwarm) # plota as probabilidade estimadas
ax.contourf(x1_surf, x2_surf, np.array(db['target_gabarito']).reshape(x1_surf.shape),
            cmap=plt.cm.coolwarm, alpha=0.5)# plota gabarito
ax.scatter(amostra.x1, amostra.x2, c=amostra.cor, marker='o', alpha=0.5, s=2) #plota os pontos


# ############################# GRAFICOS 4D  ##############################
# superficie plotada com curva teórica pintada de acordo com a probabilidade prevista
fig = plt.figure()
ax = fig.gca(projection='3d')
# logito_surf = np.array(db['logito_gabarito']).reshape(x1_surf.shape)
prob_prev_surf = np.array(db['prob_prev']).reshape(x1_surf.shape)
ax.plot_surface(x1_surf, x2_surf, prob_prev_surf, cmap=plt.cm.coolwarm)
ax.scatter(amostra['x1'], amostra['x2'], amostra['logito_gabarito'],
             c=amostra['cor'], marker='x', s=3, alpha=0.9)

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