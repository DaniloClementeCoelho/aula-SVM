import numpy as np

def min_max_funcao(beta, limite_eixos = 10):
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
    minimo = logito.min()
    maximo = logito.max()

    return minimo, maximo