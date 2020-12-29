
import numpy as np
import matplotlib.pyplot as plt

# Our 2-dimensional distribution will be over variables X and Y
N = 100
X = np.linspace(-50, 50, N)
Y = np.linspace(-50, 50, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu1 = np.array([-10., -10.])
Sigma1 = np.array([[1., -0.5], [-0.5,  1.5]])

mu2 = np.array([10., 10.])
Sigma2 = np.array([[10., 0], [0,  10.]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z1 = multivariate_gaussian(pos, mu1, Sigma1)
Z2 = multivariate_gaussian(pos, mu2, Sigma2)

# Create a surface plot and projected filled contour plot under it.
Z = Z1+Z2
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)

plt.show()