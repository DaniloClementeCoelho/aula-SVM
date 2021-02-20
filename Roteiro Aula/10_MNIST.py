import gzip
import struct
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import matplotlib.pyplot as plt
import pandas as pd

# funcao de leitura dos arquivos base
def read_idx(filename):
    with gzip.open(filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# monta base treinamento
raw_train = read_idx("./Bases/MNIST/train-images-idx3-ubyte.gz")
train_data = raw_train.reshape(60000, 28*28)
train_label = read_idx("./Bases/MNIST/train-labels-idx1-ubyte.gz")

# monta base teste
raw_test = read_idx("./Bases/MNIST/t10k-images-idx3-ubyte.gz")
test_data = raw_test.reshape(10000, 28*28)
test_label = read_idx("./Bases/MNIST/t10k-labels-idx1-ubyte.gz")

'''
amostra=base_MNIST.head()
base_MNIST = pd.DataFrame(data=amostra)
base_MNIST.to_excel("./Bases/MNIST.xlsx")
'''

X_total = np.concatenate((train_data, test_data))/255.0  # padronização pras variáveis ficarem entre 0 e 1
Y_total = np.concatenate((train_label, test_label))
base_sep = np.repeat([-1, 0], [60000, 10000])
ps = PredefinedSplit(base_sep)

'''
ps.get_n_splits()
for train_index, test_index in ps.split():
    print("TRAIN:", train_index, "TEST:", test_index)
'''


# visualiza os dados
foto = 123
fig = plt.figure(figsize=(2, 2))
ax = fig.add_subplot(111)
ax.set_axis_off()
ax.imshow(raw_train[foto, :], cmap=plt.cm.gray_r, interpolation='nearest')
ax.set_title("valor do target = " + np.str(train_label[foto]))


# diminue/filtra a base de trainamento
idx = (train_label == 2) | (train_label == 3) | (train_label == 8)
X = train_data[idx]/255.0  # padronização pras variáveis ficarem entre 0 e 1
Y = train_label[idx]

# Ajuste do modelo
classificador = svm.SVC(kernel='rbf')
parametros = {'C': [1, 10], 'gamma': [1, 0.1]}
grid = GridSearchCV(estimator=classificador, param_grid=parametros, cv=ps)
modelo = grid.fit(X_total, Y_total)
# modelo = classificador.fit(X, Y)
#### modeloT = classificador.fit(X_total, Y_total)
# modelo.cv_results_
# modelo.best_estimator_
# modelo.best_params_
# modelo.best_score_

idx = (test_label == 2) | (test_label == 3) | (test_label == 8)
x_test = test_data[idx]/255.0
y_true = test_label[idx]
y_pred = modelo.predict(x_test)
### y_pred = modeloT.predict(X_total)
### y_true = Y_total
### x_test = X_total

# matriz = metrics.confusion_matrix(Y_total , y_pred)
metrics.plot_confusion_matrix(modelo, x_test, y_true)
### metrics.plot_confusion_matrix(modeloT, X_total, Y_total)

selecao = (y_pred == 8) & (y_true == 6)
qtde_fotos_selecionadas = x_test[selecao].shape[0]
selecionadas = x_test[selecao]
fig=plt.figure(figsize=(1, 5))
for i in range(qtde_fotos_selecionadas):
    ax=fig.add_subplot(qtde_fotos_selecionadas, 1, i+1)
    ax.set_axis_off()
    ax.imshow(selecionadas[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')