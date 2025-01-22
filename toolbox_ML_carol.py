
#Carga de paquetes y módulos necesarios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, spearmanr

#1. Funcion: get_features_cat_regression

def get_features_cat_regression(df, target_col, pvalue):

    """
    Identifica columnas categóricas en un DataFrame cuya relación estadística con la columna objetivo es significativa.
    Aplica ANOVA si la columna objetivo es continua y correlación de Spearman si es discreta.

    Argumentos:
    - df (pd.DataFrame): DataFrame de entrada.
    - target_col (str): Nombre de la columna objetivo (debe ser numérica).
    - pvalue (float): Nivel de significancia estadística.

    Retorna:
    - dict: Diccionario con las columnas categóricas significativas y sus respectivos p-valores.
    """

    # Validaciones iniciales
    #Comprobar DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("La entrada no es un DataFrame válido.")
    #Comprobar si existe la columna target y que es una variable numérica.
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"La columna objetivo '{target_col}' no es numérica.")
    #Comprobar que el pvalue intruducido sea una valor entre 0 y 1
    if not (0 < pvalue < 1):
        raise ValueError("El valor de 'pvalue' debe estar entre 0 y 1.")

    # Identificar columnas categóricas (cardinalidad <10), sino devuelve {} vacía.
    cat_columns = [col for col in df.columns if col != target_col and (pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 10)]
    if not cat_columns:
        print("No se encontraron columnas categóricas en el DataFrame.")
        return {}
    
    # Identificar si target_col es continua o discreta
    is_continuous = df[target_col].nunique() > 10

    # Preparar el diccionario de columnas categóricas significativas
    significant_features = {}

    # Iterar sobre cada columna categórica
    for col in cat_columns:
        # Filtrar filas con valores nulos
        df_valid = df.dropna(subset=[col, target_col])
        if is_continuous:
            # ANOVA: Comparar medias entre grupos
            groups = [df_valid[df_valid[col] == category][target_col] for category in df_valid[col].unique()]
            if len(groups) > 1:  # Asegurarse de que haya al menos 2 grupos
                f_stat, p_val = f_oneway(*groups)
            else:
                p_val = 1  # No es posible realizar ANOVA con un solo grupo
        else:
            # Spearman: Correlación para variables discretas
            unique_categories = pd.get_dummies(df_valid[col], drop_first=True)
            p_vals = []
            for category in unique_categories.columns:
                _, p_val = spearmanr(unique_categories[category], df_valid[target_col])
                p_vals.append(p_val)
            p_val = min(p_vals)  # Usar el p-value más bajo

        # Si la relación es estadísticamente significativa, agregar al diccionario
        if p_val < pvalue:
            significant_features[col] = p_val

    return significant_features

#2. Funcion: plot_features_cat_regression

def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Identifica columnas categóricas o numéricas en un DataFrame cuya relación estadística
    con una columna objetivo ('target_col') es significativa y genera histogramas opcionales.

    Argumentos:
    - df: DataFrame de entrada.
    - target_col: Nombre de la columna objetivo, valor por defecto: "".
    - columns: Lista de columnas a analizar. Si está vacía, se analizan todas las columnas numéricas. Valor por defecto: [].
    - pvalue: nivel para la significancia estadística (por defecto su valor es 0.05).
    - with_individual_plot: Si es True, genera histogramas para las columnas significativas. Valor por defecto: False.

    Retorna:
    - list: Lista de columnas significativas según el test estadístico.
    """

    # Validaciones de entrada
    #Comprueba que es un DataFrame válido
    if not isinstance(df, pd.DataFrame):
        raise ValueError("La entrada no es un DataFrame válido.")
    #Comprobar si existe la columna target y que es una variable numérica.
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"La columna objetivo '{target_col}' no es numérica.")
    #Comprobar si el argumento columna es un strings
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise ValueError("El argumento 'columns' debe ser una lista de strings.")
    #Comprobar si el argumento "with_individual_plot" es un booleano
    if not isinstance(with_individual_plot, bool):
        raise ValueError("El argumento 'with_individual_plot' debe ser de tipo booleano.")

    # Seleccionar columnas categóricas y de baja cardinalidad si no se especifican
    if not columns:
        columns = [col for col in df.columns if col != target_col and (pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() <= 10)]

    # Filtrar filas con valores nulos una sola vez
    initial_rows = len(df)
    df = df.dropna(subset=[target_col] + columns)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Aviso: Se eliminaron {rows_dropped} filas debido a valores nulos en las columnas seleccionadas.")

    # Diccionario para almacenar resultados significativos
    significant_columns = {}

    # Iterar sobre las columnas especificadas
    for col in columns:
        if col not in df.columns:
            print(f"Aviso: La columna '{col}' no existe en el DataFrame. Se omitirá.")
            continue
        if not pd.api.types.is_categorical_dtype(df[col]) and df[col].nunique() > 10:
            print(f"Aviso: La columna '{col}' no es categórica ni tiene baja cardinalidad. Se omitirá.")
            continue

        # Aplicar ANOVA para evaluar la relación entre la columna y target_col
        groups = [df[df[col] == category][target_col] for category in df[col].unique()]
        if len(groups) > 1:  # Asegurarse de que haya al menos 2 grupos
            f_stat, p_val = f_oneway(*groups)
        else:
            p_val = 1  # No es posible realizar ANOVA con un solo grupo

        # Si la relación es significativa, agregar al diccionario
        if p_val < pvalue:
            significant_columns[col] = p_val

    # Generar gráficos si se solicita
    if with_individual_plot and significant_columns:
        num_plots = len(significant_columns)
        cols = 2
        rows = (num_plots + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        axes = axes.flatten()

        for i, (col, p_val) in enumerate(significant_columns.items()):
            for category in df[col].unique():
                subset = df[df[col] == category][target_col]
                axes[i].hist(subset, bins=bins, alpha=0.6, label=f"{col}={category}")
            axes[i].set_title(f"'{col}' vs '{target_col}' (p={p_val:.4f})")
            axes[i].set_xlabel(target_col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].legend()

        # Eliminar ejes sobrantes si hay menos gráficos que subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

    return significant_columns