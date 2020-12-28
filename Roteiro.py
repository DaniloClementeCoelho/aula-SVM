# ##################################### RODADA 1 ##############################################
# rodar pelo console interativo do Python (Alt+Shift+E)
# ##############################    INCLUI BIBLIOTECAS      ###################################
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from funções.maxima_acuracia import max_acuracia
from funções.Gera_Base import gera_base_quadrática
from funções.superficie_separacao_maxima import sup_sep_max
from funções.analise_superficie_gráfica import gera_graficos
from funções.minimo_maximo_funcao_proposta import min_max_funcao


# ##############################    SETA PARÂMETROS      ###################################
# parâmetros do hiperplano que vai gerar a superfície de classificação

limite_eixos=50  # quanto maior, maior será o tamanho da amostra e processamento

beta={  'intercepto'  : '1',
                'x1'  : '1',
                'x2'  : '1',
                'x1x2': '2',
                'x1^2': '0',
                'x2^2': '0'
        }
LI, LS = min_max_funcao(beta,limite_eixos)
print(LI, LS) #olhar os limites para colocar um "corte" que faça sentido

# ##################################### RODADA 2 ##############################################
# definiR o hiperplano de corte a ser utilizado e a variação do ruído na geração da amostra
corte=2500                  #depois de olhar os limites da função, escolher um corte que faça um desenho interessante a ser explorado
ruido= abs(corte)/3       # valor do desvio padrão do ruído, para que as regressões não acertem 100%. Escolher valor adequado olhando os gráficos

# ####################   GERA BASE, SUPERFICIE E HIPERPLANO    ####################################
db, x1_surf, x2_surf = gera_base_quadrática(corte=corte, beta=beta, limite_eixos=limite_eixos, ruido=ruido)
gera_graficos(db, x1_surf, x2_surf, corte) #olhar o gráfico e ajustar o corte e ruído, se necessário

# ##################################### RODADA 3 ##############################################
# ##############################   CONFIGURA GRÁFICOS    ######################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('x1')
ax.set_xlim(-limite_eixos, limite_eixos)
ax.set_ylabel('x2')
ax.set_ylim(-limite_eixos, limite_eixos)



# ###################    AJUSTA OS MODELOS   ######################################
amostra = db.sample(500)
ax.scatter(amostra.x1, amostra.x2, c=amostra.cor)

# MODELO LINEAR SIMPLES
logistica1 = smf.logit(formula='target ~ x1 + x2', data=amostra).fit()
logistica1.summary()
accuracy1, prob_max_ac, logito_otimo_ajustado1 = max_acuracia(logistica1)
#calcula e plota superfície de separação máxima
superficie_otima1 = sup_sep_max(db, logistica1, logito_otimo_ajustado1)
ax.scatter(superficie_otima1.x1, superficie_otima1.x2, c='black', marker=',', s=1, label=accuracy1)

# MODELO LINEAR QUADRÁTICO

logistica2 = smf.logit(formula='target ~ x1 + x2 + I(x1*x2) + I(x1*x1) + I(x2*x2)', data=amostra).fit()
logistica2.summary()
accuracy2, prob_max_ac, logito_otimo_ajustado2 = max_acuracia(logistica2)
#calcula e plota superfície de separação máxima
superficie_otima2 = sup_sep_max(db, logistica2, logito_otimo_ajustado2)
ax.scatter(superficie_otima2.x1, superficie_otima2.x2, c='black', marker=',', s=1, label=accuracy2)

plt.legend()

# plota gabarito
ax.contourf(x1_surf, x2_surf, np.array(db['target_gabarito']).reshape(x1_surf.shape),
            cmap=plt.cm.coolwarm, alpha=0.5)


# ############################# GRAFICOS 4D  ##############################
#superficie plotada com curva teórica pintada de acordo com a probabilidade prevista
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
logito_surf = np.array(db['logito_gabarito']).reshape(x1_surf.shape)
prob_prev_surf = np.array(db['prob_prev']).reshape(x1_surf.shape)
surf = ax.plot_surface(x1_surf, x2_surf, logito_surf, cmap=cm.coolwarm)

####### CONFIGURA GRÁFICOS PRA IR MOSTRANDO PONTO A PONTO
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_xlabel('x1')
ax3d.set_xlim(-limite_eixos, limite_eixos)
ax3d.set_ylabel('x2')
ax3d.set_ylim(-limite_eixos, limite_eixos)
ax3d.set_zlabel('f(x1,x2)')
#superficie plotada com curva teórica preta opaca
logito_surf = np.array(db['logito_gabarito']).reshape(x1_surf.shape)
ax3d.plot_surface(x1_surf, x2_surf, logito_surf,
                  label='superficie', color='black', alpha=0.3) #curva toda

#hiperplano escolhido para geração da amostra
hiperplano_corte = np.zeros(x1_surf.size).reshape(x1_surf.shape) + corte
ax3d.plot_surface(x1_surf, x2_surf, hiperplano_corte,
                  alpha=0.3, color = 'orange', label='hiperplano') #plano do corte

#pontos de intersecção entre superfície e hiperplano
base_pontos_plano_cortado = db[round(db['logito_gabarito']) == corte]
ax3d.scatter(base_pontos_plano_cortado['x1'], base_pontos_plano_cortado['x2'], base_pontos_plano_cortado['logito_gabarito'],
             c='black', marker='o', s=2, alpha=0.9)

#Todos os pontos da amostra no gráfico coloridos
ax3d.scatter(amostra['x1'], amostra['x2'], amostra['logito_gabarito'],
             c=amostra['cor'], marker='x', s=3, alpha=0.9)


ax3d.legend()





'''
#superficie plotada ponto a ponto e colorida com target
base_plot = db.sample(1000)
ax3d.scatter(base_plot['x1'], base_plot['x2'], base_plot['logito_gabarito'],
             c=base_plot['target'], cmap=plt.cm.coolwarm, marker='.', s=1, alpha=0.4)
'''