import numpy as np

def max_acuracia(modelo):
    acc = np.empty(100, dtype=float)
    max_acc = 0
    for i in range(1, 100, 1):
        acc[i] = (modelo.pred_table(threshold=i/100)[0,0]+modelo.pred_table(threshold=i/100)[1,1])/modelo.pred_table(threshold=i/100).sum()
        if acc[i] > max_acc:
            max_acc = acc[i]
            corte = i/100
    return max_acc, corte, acc, round(np.log((corte) / (1 - corte)), 2)

def max_sensibilidade(modelo):
    sen = np.empty(100, dtype=float)
    max_sen = 0
    for i in range(1, 100, 1):
        sen[i] = modelo.pred_table(threshold=i/100)[1, 1]/(modelo.pred_table(threshold=i/100)[1, 0]+modelo.pred_table(threshold=i/100)[1, 1])
        if sen[i] > max_sen:
            max_sen = sen[i]
            corte = i/100
    return max_sen, corte, sen, round(np.log((corte) / (1 - corte)), 2)