import numpy as np
import matplotlib.pyplot as plt

def gera_graficos(base, x1_surf, x2_surf, corte):
    base_plot = base.sample(1000)

    # da uma olhada como ficou 3D
    figura = plt.figure(figsize=(15, 15))
    ax_3D = figura.add_subplot(211, projection='3d')
    ax_3D.view_init(elev=-90., azim=0)
    ax_3D.plot_surface(
        x1_surf, x2_surf, np.array(base['logito_gabarito']).reshape(x1_surf.shape),
        alpha=0.3, color='black', label='logito')
    ax_3D.plot_surface(
        x1_surf, x2_surf, np.zeros(x1_surf.size).reshape(x1_surf.shape) + corte,
        alpha=0.3, color = 'orange', label='hiperplano classificatório')
    intercecção = base[round(base['logito_gabarito']) == corte]
    ax_3D.scatter(round(intercecção['x1']), round(intercecção['x2']), round(intercecção['logito_gabarito']),
                 c='black', marker='_', s=1, alpha=0.5, label='intercecção' )

    # da uma olhada como ficou 2D
    ax_2D = figura.add_subplot(212)
    ax_2D.scatter(base_plot['x1'], base_plot['x2'],
                  c=base_plot['cor'], marker=',', s=2, alpha=0.9)
    ax_2D.scatter(round(intercecção['x1']), round(intercecção['x2']),
                  c='black', marker='o', s=1)
    ax_2D.axis("off")
    plt.axis('equal')
    plt.tight_layout()
    return figura