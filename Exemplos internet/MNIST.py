import gzip
import struct
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import matplotlib.pyplot as plt

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

X_total = np.concatenate((train_data, test_data))/255.0
Y_total = np.concatenate((train_label, test_label))
base_sep = np.repeat([-1, 0], [60000, 10000])
ps = PredefinedSplit(base_sep)
'''
ps.get_n_splits()
for train_index, test_index in ps.split():
    print("TRAIN:", train_index, "TEST:", test_index)
'''


# visualiza os dados
foto = 23454
fig = plt.figure(figsize=(2, 2))
ax = fig.add_subplot(111)
ax.set_axis_off()
ax.imshow(raw_train[foto], cmap=plt.cm.gray_r, interpolation='nearest')
ax.set_title("valor do target = " + np.str(train_label[foto]))

idx = (train_label == 2) | (train_label == 3) | (train_label == 8)
X = train_data[idx]/255.0
Y = train_label[idx]

# Ajuste do modelo
classificador = svm.SVC(kernel='rbf')
parametros = {'C': [0.1, 1, 10], 'gammma': [0.1, 0.01, 0.001]}
grid = GridSearchCV(estimator=classificador, param_grid=parametros, cv=ps)
modelo = grid.fit(X_total, Y_total)
# modelo = classificador.fit(X, Y)


idx = (test_label == 2) | (test_label == 3) | (test_label == 8)
x_test = test_data[idx]/255.0
y_true = test_label[idx]
y_pred = modelo.predict(x_test)

# matriz = metrics.confusion_matrix(y_true, y_pred)
metrics.plot_confusion_matrix(modelo, x_test, y_true)

selecao = (y_pred == 8) & (y_true == 3)
qtde_fotos_selecionadas = x_test[selecao].shape[0]
fig=plt.figure(figsize=(1, 5))
for i in range(qtde_fotos_selecionadas):
    ax=fig.add_subplot(qtde_fotos_selecionadas, 1, i+1)
    ax.set_axis_off()
    ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')











X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
test_fold = [0, 1, -1, 1]
ps = PredefinedSplit(test_fold)
ps.get_n_splits()

print(ps)

for train_index, test_index in ps.split():
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]