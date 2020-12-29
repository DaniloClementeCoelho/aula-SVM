import pandas as pd
import numpy as np

# beta = vetor de parâmetros desejados ("gabarito")
# hiperplano que vai gerar a superfície de classificação


def gera_base(x1, x2, logito, corte, ruido):

    # inicializa variáveis do target e cor
    target = np.empty(len(x1), dtype=int)
    target_real = np.empty(len(x1), dtype=int)
    cor = np.empty(len(x1), dtype=object)
    noise = ruido  # valor da aletorização - ruído não explicado pelo modelo

    # simula reposta aleatória
    for i in range(len(logito)):
        if logito[i] + np.random.normal(0, noise) > corte:
            target[i] = 1
            cor[i] = 'red'
        else:
            target[i] = 0
            cor[i] = 'blue'
    # calcula o valor real sem ruido para gráfico de contorno real
    for i in range(len(logito)):
        if logito[i] > corte:
            target_real[i] = 1
        else:
            target_real[i] = 0

    # cria uma base para modelo
    base = pd.DataFrame({'x1': x1, 'x2': x2, 'target': target, 'cor': cor,
                         'logito_gabarito': logito, 'target_gabarito': target_real})

    return base
