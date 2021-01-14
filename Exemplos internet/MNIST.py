import gzip
import struct
import numpy as np
from sklearn import svm, metrics
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

modelo = svm.SVC(C=5, gamma=0.05).fit(X, Y)

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
