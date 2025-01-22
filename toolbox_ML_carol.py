#Funcion: get_features_cat_regression
#Carga de paquetes y módulos necesarios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, spearmanr

#1. Funcion: get_features_cat_regression

def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Función que recibe las columnas categóricas en un DataFrame cuya relación estadística con la columna objetivo
    ('target_col') es significativa, basándose en el test estadístico apropiado. (ANOVA, Spearman)

    Argumentos:
    df: DataFrame de entrada.
    target_col: Nombre de la columna objetivo (debe ser numérica(str)).
    pvalue: nivel para la significancia estadística (por defecto su valor es 0.05).

    Retorna:
    list: lista con las columnas categóricas del dataframe cuyo test de relación con la columna designada por 'target_col'
    tenga una relación significativa con la columna objetivo, o None si las entradas son inválidas.

    """
    # Comprobaciones necesarias:
    #Validación de los argumentos
    # 1. ¿Carga del df correcto?
    if not isinstance(df, pd.DataFrame):
        print("Error: La entrada no es un DataFrame de Pandas, por favor revise el archivo.")
        return None
    # 2. Columnas: ¿La valiable es numérica?
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: La columna objetivo '{target_col}' no es numérica, por favor, revise los datos.")
        return None

    #Identificar columnas categóricas en el DataFrame, sino devuelve lista vacía
    cat_columns = [col for col in df.columns if col != target_col and pd.api.types.is_categorical_dtype(df[col])]
    if not cat_columns:
        print("No se encontraron columnas categóricas en el DataFrame.")
        return []

    # Lista para almacenar las columnas categóricas significativas
    significant_features = []

    # Identificar si target_col es continua o discreta con alta cardinalidad
    if df[target_col].nunique() > 10:  # Más de 10 valores únicos se considera alta cardinalidad/continua
        is_continuous = True
    else:
        is_continuous = False

    # Iterar sobre las columnas categóricas y aplicar el test adecuado
    for col in cat_columns:
        # Comprobar que no haya valores nulos en la columna categórica y en target_col
        if df[col].isnull().any() or df[target_col].isnull().any():
            print(f"Aviso: La columna '{col}' contiene valores nulos.")
            df = df.dropna(subset=[col, target_col])

        # Si la columna objetivo es continua, usar ANOVA
        if is_continuous:
            groups = [df[df[col] == category][target_col] for category in df[col].unique()]
            f_stat, p_val = f_oneway(*groups)
        else:
            # Si la columna objetivo es discreta, usar correlación de Spearman
            unique_categories = pd.get_dummies(df[col], drop_first=True)
            correlations = []
            for category in unique_categories.columns:
                spearman_corr, p_val = spearmanr(unique_categories[category], df[target_col])
                correlations.append(p_val)

            # Tomar el mínimo p-value para decidir si la relación es significativa
            p_val = min(correlations)

        # Si el p-value es menor que el umbral, agregar la columna a la lista de significativas
        if p_val < pvalue:
            significant_features.append(col)

    return significant_features

#2. Funcion: plot_features_cat_regression

def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Identifica columnas categóricas o numéricas en un DataFrame cuya relación estadística
    con una columna objetivo ('target_col') es significativa y genera histogramas opcionales.

    Argumentos:
    df: DataFrame de entrada.
    target_col: Nombre de la columna objetivo, valor por defecto: "".
    columns: Lista de columnas a analizar. Si está vacía, se analizan todas las columnas numéricas. Valor por defecto: [].
    pvalue: nivel para la significancia estadística (por defecto su valor es 0.05).
    with_individual_plot: Si es True, genera histogramas para las columnas significativas. Valor por defecto: False.

    Retorna:
    list: Lista de columnas significativas según el test estadístico.
    """

    # Validaciones de entrada
    # 1. ¿Carga del df correcto?
    if not isinstance(df, pd.DataFrame):
        print("Error: La entrada no es un DataFrame de Pandas. Por favor, revise los datos.")
        return None
    # 2. ¿Es una columna numérica?
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: La columna objetivo '{target_col}' no es numérica. Por favor, revise los datos.")
        return None
    #3. Formato correcto del argumento columna
    if not isinstance(columns, list):
        print("Error: El argumento 'columns' debe ser una lista de strings.")
        return None
    #4. Formato correcto de los elementos de la columna
    if not all(isinstance(col, str) for col in columns):
        print("Error: Todos los elementos de 'columns' deben ser strings.")
        return None
    #5. Formato del argumento "with_individual_plot"
    if not isinstance(with_individual_plot, bool):
        print("Error: El argumento 'with_individual_plot' debe ser de tipo booleano.")
        return None

    # Si la lista de columnas está vacía, seleccionar todas las columnas categóricas
    if not columns:
        columns = [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]

    # Almacenar las columnas categóricas significativas
    significant_columns = []

    # Iterar sobre las columnas especificadas
    for col in columns:
        if col not in df.columns:
            print(f"Aviso: La columna '{col}' no existe en el DataFrame. Se omitirá.")
            continue

        if not pd.api.types.is_categorical_dtype(df[col]):
            print(f"Aviso: La columna '{col}' no es categórica. Se omitirá.")
            continue

        # Comprobar valores nulos
        if df[col].isnull().any() or df[target_col].isnull().any():
            print(f"Aviso: La columna '{col}' contiene valores nulos. Se ignorarán estas filas.")
            df = df.dropna(subset=[col, target_col])

        # Aplicar ANOVA para evaluar la relación entre la columna y target_col
        groups = [df[df[col] == category][target_col] for category in df[col].unique()]
        f_stat, p_val = f_oneway(*groups)

        # Si el p-value es menor que el umbral, agregar la columna a la lista significativa
        if p_val < pvalue:
            significant_columns.append(col)

            # Si se solicita, generar histogramas
            if with_individual_plot:
                plt.figure(figsize=(8, 6))
                for category in df[col].unique():
                    subset = df[df[col] == category][target_col]
                    plt.hist(subset, bins=10, alpha=0.6, label=f"{col}={category}")
                plt.title(f"Distribución de '{target_col}' para '{col}'")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.legend()
                plt.show()

    return significant_columns