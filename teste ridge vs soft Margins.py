import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

base = pd.read_excel("./Bases/teste_basico_SVM_ridgeXsoft.xlsx")
explicativas = base.iloc[:, 0:2]

SVM = svm.SVC(kernel='rbf', C=0.6, gamma=0.1)
# SVM = svm.SVC(kernel='linear')
SVM.fit(explicativas, base.target)

x1_surf, x2_surf = np.meshgrid(np.arange(-1, 11, 0.1),
                               np.arange(-1, 11, 0.1))
x1 = x1_surf.ravel()
x2 = x2_surf.ravel()
X = np.concatenate([x1.reshape(x1.size, 1), x2.reshape(x1.size, 1)], axis=1)
prev_svm = SVM.predict(X).reshape(x1_surf.shape)

figura = plt.figure()
ax = figura.add_subplot(111)
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.scatter(SVM.support_vectors_[:,0], SVM.support_vectors_[:,1], facecolors='none', edgecolors='black', s=100)
ax.scatter(base['x2'], base['x1'], c=base['cor'])
ax.contourf(x1_surf, x2_surf, prev_svm, cmap=plt.cm.coolwarm, alpha=0.3)
plt.show()