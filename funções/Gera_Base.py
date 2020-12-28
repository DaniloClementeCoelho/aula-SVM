import pandas as pd
import numpy as np

# beta = vetor de parâmetros desejados ("gabarito")
# hiperplano que vai gerar a superfície de classificação

def gera_base_quadrática(beta, corte, limite_eixos = 10):
    x1_surf, x2_surf = np.meshgrid(np.arange(-limite_eixos, limite_eixos, 0.1),
                       np.arange(-limite_eixos, limite_eixos, 0.1))
    x1 = x1_surf.ravel()
    x2 = x2_surf.ravel()

    # função a ser estudada
    beta0 = float(beta['intercepto'])
    beta1 = float(beta['x1'])
    beta2 = float(beta['x2'])
    beta12 = float(beta['x1x2'])
    beta11 = float(beta['x1^2'])
    beta22 = float(beta['x2^2'])
    logito = np.around(beta0 + beta1*x1 + beta2*x2 + beta12*x1*x2 + beta11*x1**2 + beta22*x2**2, 2)

    # inicializa variáveis do target e cor
    target = np.empty(len(x1), dtype=int)
    target_real = np.empty(len(x1), dtype=int)
    cor = np.empty(len(x1), dtype=object)
    noise = logito.mean()/3 #valor da aletorização - ruído não explicado pelo modelo

    #simula reposta aleatória
    for i in range(len(logito)):
        if logito[i] + np.random.normal(0, noise) > corte:
            target[i] = 1
            cor[i] = 'red'
        else:
            target[i] = 0
            cor[i] = 'blue'
    #calcula o valor real sem ruido para gráfico de contorno real
    for i in range(len(logito)):
        if logito[i] > corte:
            target_real[i] = 1
        else:
            target_real[i] = 0

    # cria uma base para modelo
    base = pd.DataFrame({'x1': x1, 'x2': x2, 'target': target, 'cor': cor,
                         'logito_gabarito': logito, 'target_gabarito': target_real })

    return base, x1_surf, x2_surf