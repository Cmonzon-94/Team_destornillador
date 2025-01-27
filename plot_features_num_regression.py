import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_features_num_regression(dataframe, target_col, columns=[], umbral_corr=0, pvalue=None):
    """
    Generar pairplot de columnas numéricas correlacionadas con la variable objetivo.
    Parámetros:
    - dataframe: DataFrame de entrada.
    - target_col: Nombre de la columna objetivo.
    - columns: Lista de columnas a evaluar (opcional).
    - umbral_corr: Umbral mínimo de correlación en valor absoluto (por defecto 0).
    - pvalue: Valor p para pruebas de hipótesis (opcional).

    Devuelve:
    - Lista de columnas numéricas seleccionadas para el gráfico.
    """
    if not columns:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    columnas_validas = get_features_num_regression(dataframe, target_col, umbral_corr, pvalue)

    columnas_a_graficar = [col for col in columns if col in columnas_validas]

    if len(columnas_a_graficar) > 5:
        for i in range(0, len(columnas_a_graficar), 5):
            sns.pairplot(dataframe, vars=[target_col] + columnas_a_graficar[i:i + 5])
            plt.show() 
    else:
        sns.pairplot(dataframe, vars=[target_col] + columnas_a_graficar)
        plt.show()  

    return columnas_a_graficar