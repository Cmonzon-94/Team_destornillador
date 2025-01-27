import pandas as pd
import numpy as np
from scipy.stats import pearsonr, f_oneway

def get_features_num_regression(dataframe, target_col, umbral_corr, pvalue=None):
    """
    Seleccionar columnas numéricas correlacionadas con la variable objetivo.
    Parámetros:
    - dataframe: DataFrame de entrada.
    - target_col: Nombre de la columna objetivo.
    - umbral_corr: Umbral mínimo de correlación en valor absoluto.
    - pvalue: Valor p para pruebas de hipótesis (opcional).

    Devuelve:
    - Lista de columnas numéricas correlacionadas con la columna objetivo.
    """

    if target_col not in dataframe.columns or not np.issubdtype(dataframe[target_col].dtype, np.number):
        print("La columna objetivo no es válida o no es numérica continua.")
        return None

    correlaciones = [] 

    for columna in dataframe.select_dtypes(include=[np.number]).columns:
        if columna != target_col:  
            corr, p = pearsonr(dataframe[columna].dropna(), dataframe[target_col].dropna())
            if abs(corr) >= umbral_corr and (pvalue is None or p <= pvalue):
                correlaciones.append(columna)

    return correlaciones 