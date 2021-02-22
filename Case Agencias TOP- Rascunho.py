# código pra aula se encontra no Jupyter


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split as tts
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

base = pd.read_excel("D:\GoogleDrive\Curso ASN.Rocks\SVM\programas\Bases\Case_Agencias.xlsx")

base.columns

perdas = base.columns[[x.startswith("perda") for x in base.columns]]
custos = base.columns[[x.startswith("custos") for x in base.columns]]
churn = base.columns[[x.startswith("churn") for x in base.columns]]
potRegiao = base.columns[[x.startswith("potencialRegiao") for x in base.columns]]
Over90 = base.columns[[x.startswith("Over90") for x in base.columns]]
PercFraudes = base.columns[[x.startswith("percFraudes") for x in base.columns]]


vars_invers_prop = base[perdas | custos | churn | potRegiao | Over90 | PercFraudes]
vars_diret_prop = base.drop(columns=perdas | custos | churn | potRegiao | Over90 | PercFraudes)


vars_invers_prop_scaled = MinMaxScaler().fit_transform(vars_invers_prop)
vars_diret_prop_scaled = MinMaxScaler().fit_transform(vars_diret_prop.iloc[:,1:271])

#Modeling
kmeans = KMeans(5)
kfit = kmeans.fit(vars_invers_prop_scaled)
base_invers_prop_cluster = kfit.predict(vars_invers_prop_scaled)
kfit.cluster_centers_
unique, counts = np.unique(kfit.labels_, return_counts=True)
# cluster 3 é o melhor
kmeans = KMeans(5)
kfit2 = kmeans.fit(vars_diret_prop_scaled)
base_diret_prop_cluster = kfit2.predict(vars_diret_prop_scaled)
kfit2.cluster_centers_
unique, counts = np.unique(kfit2.labels_, return_counts=True)
# cluster 0 é o melhor

base['Cl_invers'] = base_invers_prop_cluster
base['Cl_diret'] = base_diret_prop_cluster
base['target'] = np.zeros(len(base['Cl_invers']))
for i in range(len(base['Cl_invers'])):
    if ( (base['Cl_invers'][i]==3 or base['Cl_invers'][i]==4 ) and (base['Cl_diret'][i]==0 or base['Cl_diret'][i]==1) ):
        base.loc[i, 'target'] = 1
    else:
        base.loc[i, 'target'] = 0


base['target'].value_counts()


# SVM
X = base.drop(columns=['id', 'Cl_invers', 'Cl_diret', 'target'])
X_norm = MinMaxScaler().fit_transform(X)
Y = base['target']


# X_train, X_test, Y_train, Y_test = tts(X_norm, Y, test_size=0.2)

classificador = svm.SVC()
modelo_ajustado = classificador.fit(X_norm, Y)
predito = classificador.predict(X_norm)

plot_confusion_matrix(classificador, X_norm, Y, values_format='d', display_labels=["demais", "TOP"])