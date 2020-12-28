import numpy as np
import matplotlib.pyplot as plt

def gera_graficos(base, x1_surf, x2_surf):
    base_plot = base.sample(1000)

    # da uma olhada como ficou 3D
    figura = plt.figure()
    ax_3D = figura.add_subplot(211, projection='3d')
    ax_2D = figura.add_subplot(212)
    ax_3D.plot_surface(
        x1_surf, x2_surf, np.array(base['logito_gabarito']).reshape(x1_surf.shape),
        alpha=0.3, color='black', label='superficie')
    ax_3D.plot_surface(
        x1_surf, x2_surf, np.zeros(x1_surf.size).reshape(x1_surf.shape) + corte,
        alpha=0.3, color = 'orange', label='hiperplano')
    intercecção = db[round(db['logito_gabarito']) == corte]
    ax_3D.scatter(intercecção['x1'], intercecção['x2'], intercecção['logito_gabarito'],
                 c='black', marker='_', s=1, alpha=0.5, label='intercecção' )
    figura.legend()

    # da uma olhada como ficou 2D
    ax_2D.scatter(base_plot['x1'], base_plot['x2'], c=base_plot['cor'])
    ax_2D.scatter(intercecção['x1'], intercecção['x2'],
                  c='black', marker=',', s=1, alpha=0.5, label='intercecção')

    return figura