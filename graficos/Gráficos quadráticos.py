import numpy as np
import matplotlib.pyplot as plt

limite=100
x1, x2 = np.meshgrid(np.linspace(-limite, limite, 1000),
                     np.linspace(-limite, limite, 1000))
x12 = x1*x2
x11 = x1**2
x22 = x2**2

beta0 = 0
beta1 = 0
beta2 = 0
beta12 = 0
beta11 = 1
beta22 = 1

Y= beta0 + beta1*x1 + beta2*x2 + beta12*x12 + beta11*x11 + beta22*x22

threshold = 12500

################### GRAFICO 3D Com corte threshold ############################
x1_v = x1[0,:]

k = ( beta12*x1_v+beta2 ) / ( 2*beta22**(1/2) )

x2_f_x1_1 = (  ( threshold - (beta0 + beta1*x1_v + beta11*x1_v**2) + k**2 )**(1/2) - k  ) / (beta22)**(1/2)
x2_f_x1_2 = (  -( threshold - (beta0 + beta1*x1_v + beta11*x1_v**2) + k**2 )**(1/2) - k  ) / (beta22)**(1/2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.xlabel('x1')
plt.ylabel('x2')

ax.plot_surface(x1, x2, Y, color='blue', alpha=0.3)
ax.plot(x1_v, x2_f_x1_1, threshold, c='red',label=threshold)
ax.plot(x1_v, x2_f_x1_2, threshold, c='red')
plt.legend()
plt.show()

################### CORTES do threshold 2D ############################
plt.gca().set_xlim([-limite, limite])
plt.gca().set_ylim([-limite, limite])

plt.plot(x1_v, x2_f_x1_1, c='blue',label=threshold)
plt.plot(x1_v, x2_f_x1_2, c='blue')
plt.gca().set_aspect('equal')
plt.show()
