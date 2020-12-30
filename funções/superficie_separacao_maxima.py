import numpy as np

def sup_sep_max(base, modelo, logito_corte):
    base['prob_prev'] = modelo.predict(base)
    base['logito_prev'] = round( np.log( (base['prob_prev'])/(1-(base['prob_prev'])) ),2)
    superficie_otima = base[base['logito_prev']==logito_corte]

    return superficie_otima