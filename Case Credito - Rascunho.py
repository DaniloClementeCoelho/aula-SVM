# código pra aula se encontra no Jupyter

import pandas as pd
import numpy as np
from sklearn import svm,
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix

base_Total = pd.read_csv("D:\GoogleDrive\Curso ASN.Rocks\Monitoramento de Modelos\envios\envio_case_base_Total.csv")
# base_Total.shape

base_Total = base_Total[(base_Total['contratado'] == 1) & (base_Total['ano_mes'] < 201911)]
# base_Total.columns

base_Total['compr_renda'] = base_Total.pmt / base_Total.renda
base_Total.drop(['id', 'data', 'FPD', 'score', 'contratado', 'pmt'], axis=1, inplace=True)


def bivariadas(variavel):
    a = pd. DataFrame()
    a ['count'] = base_Total[variavel].value_counts()
    a ['percent'] = base_Total[variavel].value_counts(normalize=True)
    a ['inad'] = base_Total.groupby(variavel).mean()['target']
    a.sort_values(by='inad', ascending=False, inplace=True)
    return a
def percentual(k):
  return k/k[-1]

bivariadas('idade_veiculo')
pd.crosstab(base_Total["idade_veiculo"], base_Total["target"], margins=True).apply(percentual, axis=1)
base_Total.pivot_table(values=["target"], index=["ano_mes", "idade_veiculo"], aggfunc=[np.mean, len], margins=True)

bivariadas('marca')
bivariadas('vl_bem')
bivariadas('estado_civil')
bivariadas('qt_restr')
bivariadas('Regiao')
bivariadas('Profissao')
bivariadas('entrada')
bivariadas('prazo')

base_Total['idade'].describe()
base_Total.boxplot(column="idade", by="target")
base_Total.hist(column="idade", by="target", bins=30)
base_Total['idade_cat'] = pd.qcut(base_Total['idade'], 5)
bivariadas('idade_cat')

base_Total['renda'].describe()
base_Total['entrada'].describe()
base_Total['pmt'].describe()
base_Total['compr_renda'].describe()

base_Total['target'].value_counts()



base_Total_dummies = pd.get_dummies(base_Total,
    columns=['idade_veiculo', 'marca', 'estado_civil', 'qt_restr','Regiao', 'Profissao', 'prazo', 'vl_bem'],
    drop_first=True)

# base_Total_dummies.columns

explicativas=['idade', 'renda', 'entrada', 'compr_renda',
        'idade_veiculo_0', 'idade_veiculo_1_2', 'idade_veiculo_3_5', 'idade_veiculo_6_10',
        'marca_Fiat', 'marca_Ford', 'marca_Honda','marca_Hyundai', 'marca_Jac', 'marca_Nissan', 'marca_Outros',
        'marca_Peugeot', 'marca_Toyota', 'marca_VW',
        'estado_civil_divorciado', 'estado_civil_solteiro', 'estado_civil_viuvo',
        'qt_restr_1', 'qt_restr_2', 'qt_restr_>2',
        'Regiao_N', 'Regiao_NE', 'Regiao_S', 'Regiao_SE',
        'Profissao_assalariados', 'Profissao_autonomos', 'Profissao_consignados',
        'Profissao_empresarios', 'Profissao_liberais', 'Profissao_prof_serv',
        'prazo_24', 'prazo_36', 'prazo_48', 'prazo_60',
        'vl_bem_(25000, 70000]', 'vl_bem_(70000, 999999]']

X = base_Total_dummies[explicativas]
X_norm = MinMaxScaler().fit_transform(X)
Y = base_Total['target']

X_train, X_test, Y_train, Y_test = tts(X_norm, Y, test_size=0.2)

grade_param = {'C': [0.1, 0.5, 1, 10, 100, 1000],
               'gamma': ['scale', 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
               }
param_otimos = GridSearchCV(svm.SVC(), grade_param, cv=3, scoring='roc_auc', verbose=2)
modelo_ajustado = param_otimos.fit(X_train, Y_train)
print(param_otimos.best_params_, "  AUC = " + str(round(param_otimos.best_score_*100, 1))+"%")


classificador = svm.SVC(kernel='rbf', C=1, gamma='scale')
modelo_ajustado = classificador.fit(X_train, Y_train)
predito = classificador.predict(X_test)

print(classification_report(Y_test, predito))
plot_confusion_matrix(classificador, X_test, Y_test, values_format='d', display_labels=["em dia", "inadimplente"])
# gridsearch
# K-means clustering
# binning
# fazendo dummmies
# padronizando escala
# matriz confusão