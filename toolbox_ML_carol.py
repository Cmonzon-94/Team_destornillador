#Funcion: get_features_cat_regression
#Carga de paquetes y módulos necesarios
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, spearmanr
#Definimos una función para encontrar variables categóricas significativas
def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Función que recibe las columnas categóricas en un DataFrame cuya relación estadística con la columna objetivo
    ('target_col') es significativa, basándose en el test estadístico apropiado.

    Argumentos:
    df: DataFrame de entrada.
    target_col: Nombre de la columna objetivo (debe ser numérica(str)).
    pvalue: Umbral para la significancia estadística (por defecto es 0.05).

    Retorna:
    list: lista con las columnas categóricas del dataframe cuyo test de relación con la columna designada por 'target_col'
    tenga una relación significativa con la columna objetivo, o None si las entradas son inválidas.

    """
    # Comprobaciones necesarias:
    # Validación de los argumentos
    # 1. ¿Carga del df correcto?
    if not isinstance(df, pd.DataFrame):
        print("Error: La entrada no es un DataFrame de pandas, por favor revise el archivo.")
        return None
    # 2. Columnas: ¿pertenecen al DF?, ¿variable categórica?
    if target_col not in df.columns:
        print(f"Error: La columna objetivo '{target_col}' no existe en el DataFrame, por favor revise el nombre de la columna.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: La columna objetivo '{target_col}' no es numérica, por favor, revise los datos.")
        return None
    # 3. pvalue= 0-1
    if not isinstance(pvalue, (float, int)) or not (0 < pvalue < 1):
        print("Error: El valor de 'pvalue', por favor, introduce un número comprendido entre el 0 y 1.")
        return None

    # Identificar columnas categóricas


