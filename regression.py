import numpy as np
#################################### CRIA A BASE ##########################################
n = 1000
sigma_error = 50

X1 = np.random.normal(10, 10, n)
X2 = np.random.normal(10, 10, n)
Y = 3.5 + 1.1*X1 + 2.2*X2 + np.random.normal(0, sigma_error, n)

target = np.empty(n, dtype=object)
for i in range(len(Y)):
    if Y[i] < 3.5 + 1.1*X1[i] + 2.2*X2[i] + np.random.normal(0, sigma_error/5, 1):
        target[i] = 'b'
    else: target[i] = 'g'


#################################### AJUSTA REGRESSÃO LINEAR ##########################################
from sklearn.linear_model import LinearRegression

X = np.concatenate([X1.reshape(n,1),X2.reshape(n,1)], axis=1)
reg_linear = LinearRegression().fit(X,Y)

#reg_linear.score(X, Y) #R^2
#reg_linear.coef_
#reg_linear.intercept_

#################################### MONTA GRÁFICO ##########################################
import matplotlib.pyplot as plt


x1_surf, x2_surf = np.meshgrid(
    np.linspace(X1.min(), X1.max(), 10),
    np.linspace(X2.min(), X2.max(), 10)
)

#y_surf=3.5 + 1.1*x1_surf + 2.2*x2_surf

X_surf = np.concatenate([x1_surf.reshape(x1_surf.size,1), x2_surf.reshape(x2_surf.size,1)], axis=1)

fittedY = reg_linear.predict(X_surf).reshape(x1_surf.shape)

figura = plt.figure()
ax = figura.add_subplot(111, projection='3d')

ax.scatter(X1, X2, Y, c=target)
#ax.plot_surface(x1_surf,x2_surf,y_surf)
ax.plot_surface(x1_surf,x2_surf,fittedY)

plt.show()

'''
type(y_surf)
type(fittedY)
y_surf.shape
fittedY.shape


X_surf.shape
'''